"""Linear SVM classification for the traffic-sign feature vectors.

This module consumes the overlap-free ``tsrd_training_features.npz`` artifact
produced with ``featureExtraction.py``. It does not perform segmentation or
feature extraction. The three-digit filename prefix is the zero-based dataset
class, so prefix 000 is reported as Sign 1.

Run from the repository root:
    python -m app.services.svmClassification
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services import featureExtraction as feature_extraction


DEFAULT_FEATURE_FILE = (
    Path(__file__).resolve().parent.parent
    / "result"
    / "tsrd_training_features.npz"
)
DEFAULT_RESULT_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "image" / "ColorInputs"
)
DEFAULT_MODEL_FILE = (
    Path(__file__).resolve().parent.parent / "models" / "svm_normal_model.npz"
)
RESULT_GROUPS = ("BlueSigns", "RedSigns", "YellowSigns")
LABEL_PATTERN = re.compile(r"(?:^|[\\/])(\d{3})(?:_|$)")

# One-based TSRD identifiers, checked against local images/000..057 examples.
# Descriptive English names; not a replacement for jurisdiction-specific rules.
SIGN_NAMES = dict(enumerate((
    "Speed limit 5 km/h", "Speed limit 15 km/h", "Speed limit 30 km/h",
    "Speed limit 40 km/h", "Speed limit 50 km/h", "Speed limit 60 km/h",
    "Speed limit 70 km/h", "Speed limit 80 km/h",
    "No straight ahead or left turn", "No straight ahead or right turn",
    "No straight ahead", "No left turn", "No left or right turn",
    "No right turn", "No overtaking", "No U-turn", "No motor vehicles",
    "No sounding horn", "End of 40 km/h restriction", "End of 50 km/h restriction",
    "Straight ahead or right", "Straight ahead", "Turn left",
    "Turn left or right", "Turn right", "Keep left", "Keep right",
    "Roundabout", "Motor vehicles only", "Sound horn", "Cycles only",
    "U-turn permitted", "Two-way traffic", "Traffic signals ahead",
    "Other danger", "Pedestrians ahead", "Cyclists ahead", "Children ahead",
    "Double bend - first right", "Double bend - first left",
    "Road narrows on left", "Steep descent", "Slow down",
    "Side road junction on right", "Side road junction on left", "Village ahead",
    "Reverse bends ahead", "Railway crossing without gates", "Roadworks",
    "Series of bends", "Railway crossing with gates", "Accident ahead",
    "Stop", "No vehicles", "No stopping", "No entry", "Give way",
    "Checkpoint",
), start=1))


def sign_description(label: int) -> str:
    """Keep the trained identifier while presenting its human-readable meaning."""
    label = int(label)
    return f"{SIGN_NAMES.get(label, 'Unknown sign')} (Sign {label})"


def add_display_arguments(parser: argparse.ArgumentParser) -> None:
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", action="store_true", dest="show",
                         help="browse original/segmented predictions (default); any key next, Esc quits")
    display.add_argument("--no-show", action="store_false", dest="show",
                         help="run terminal evaluation without opening image windows")
    parser.set_defaults(show=True)
    parser.add_argument("--image", type=Path,
                        help="show a single source image after the standard evaluation")


def render_prediction(original: np.ndarray, segmented: np.ndarray | None,
                      caption: str, filename: str) -> np.ndarray:
    """Build an original/segmented comparison without opening a window."""
    canvas = np.zeros((410, 960, 3), dtype=np.uint8)
    for offset, picture in ((0, original), (480, segmented)):
        if picture is None or picture.size == 0:
            continue
        height, width = picture.shape[:2]
        ratio = min(460 / width, 300 / height)
        resized = cv2.resize(picture, (max(1, round(width * ratio)),
                                      max(1, round(height * ratio))))
        h, w = resized.shape[:2]
        x, y = offset + (480-w)//2, (310-h)//2
        canvas[y:y+h, x:x+w] = resized
    def text_line(text, position, max_width):
        scale = 0.65
        while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_width:
            scale *= 0.9
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255,255,255), 1, cv2.LINE_AA)
    text_line("Original", (12,338), 456)
    text_line(caption, (492,338), 456)
    text_line(filename, (12,373), 936)
    text_line("Any key: next image    Esc: close", (12,400), 936)
    return canvas


def display_predictions(model, extractor, args) -> None:
    """Optional GUI: names come only from model predictions, never filenames."""
    if not args.show and args.image is None:
        return
    paths = ([args.image] if args.image is not None else
             [p for group in RESULT_GROUPS
              for p in sorted((args.result_directory / group).glob("*.png"))])
    title = "Traffic sign detection"
    try:
        for path in paths:
            original = cv2.imread(str(path))
            if original is None:
                raise ValueError(f"Cannot read image: {path}")
            segmented = None
            caption = "Unable to extract sign features"
            try:
                segmented = feature_extraction.segment_traffic_sign(path)
                vector = extractor.extract_features(path)
                if vector is not None:
                    caption = sign_description(model.predict(vector)[0])
            except (ValueError, cv2.error):
                pass
            print(f"{path.name}: {caption}", flush=True)
            cv2.imshow(title, render_prediction(original, segmented, caption, path.name))
            if cv2.waitKey(0) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()


def sign_label_from_path(image_path: str) -> int:
    """Convert a stored filename prefix to the confirmed one-based sign ID."""
    match = LABEL_PATTERN.search(str(image_path))
    if match is None:
        raise ValueError(f"Cannot find a three-digit sign label in: {image_path}")
    return int(match.group(1)) + 1


def load_feature_dataset(feature_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the feature matrix and derive sign labels from saved image paths."""
    with np.load(feature_file, allow_pickle=False) as archive:
        required = {"X", "image_paths"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Feature file is missing arrays: {sorted(missing)}")
        features = np.asarray(archive["X"], dtype=np.float32)
        image_paths = np.asarray(archive["image_paths"]).reshape(-1)

    if features.ndim != 2 or features.shape[0] != image_paths.size:
        raise ValueError("X must contain one two-dimensional feature row per image path")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Feature matrix is empty")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinite values")

    labels = np.asarray(
        [sign_label_from_path(str(path)) for path in image_paths],
        dtype=np.int32,
    )
    return features, labels


def stratified_holdout(
    labels: np.ndarray,
    validation_ratio: float = 0.2,
    seed: int = 2513,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return non-overlapping train/validation indices for the small dataset.

    Classes with one example remain training-only because they cannot be
    represented in both sets. Every class with two or more examples contributes
    at least one validation item.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    singleton_labels: list[int] = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = indices[rng.permutation(indices.size)]
        if indices.size == 1:
            train_indices.extend(indices.tolist())
            singleton_labels.append(int(label))
            continue
        validation_count = min(
            indices.size - 1,
            max(1, int(round(indices.size * validation_ratio))),
        )
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())

    return (
        np.asarray(train_indices, dtype=np.int32),
        np.asarray(validation_indices, dtype=np.int32),
        singleton_labels,
    )


def balance_training_classes(
    features: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int = 160,
    seed: int = 2513,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic training set with equal class frequencies."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Features and labels have incompatible shapes")

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        chosen = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=class_indices.size < samples_per_class,
        )
        selected_indices.extend(chosen.tolist())

    selected = np.asarray(selected_indices, dtype=np.int32)
    selected = selected[rng.permutation(selected.size)]
    return features[selected], labels[selected]


@dataclass
class LinearSVM:
    """One-vs-rest linear Support Vector Machine trained with hinge loss."""

    labels: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def predict(self, feature_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(feature_rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"Expected feature rows of length {self.weights.shape[1]}, "
                f"received shape {rows.shape}"
            )
        standardized = (rows - self.feature_mean) / self.feature_scale
        scores = standardized @ self.weights.T + self.bias
        return self.labels[np.argmax(scores, axis=1)]

    def predict_sign(self, feature_vector: np.ndarray) -> str:
        return sign_description(self.predict(feature_vector)[0])


def save_linear_svm(
    model: LinearSVM,
    model_file: str | Path = DEFAULT_MODEL_FILE,
    feature_extractor: str = "featureExtraction.py",
) -> Path:
    """Save a trained linear SVM as a compressed, pickle-free NPZ archive."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_file,
        format_version=np.asarray(1, dtype=np.int32),
        model_kind=np.asarray("linear_svm"),
        feature_extractor=np.asarray(feature_extractor),
        labels=np.asarray(model.labels, dtype=np.int32),
        weights=np.asarray(model.weights, dtype=np.float32),
        bias=np.asarray(model.bias, dtype=np.float32),
        feature_mean=np.asarray(model.feature_mean, dtype=np.float32),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float32),
    )
    return model_file


def load_linear_svm(
    model_file: str | Path = DEFAULT_MODEL_FILE,
    expected_feature_extractor: str = "featureExtraction.py",
) -> LinearSVM:
    """Load and validate a linear SVM saved by :func:`save_linear_svm`."""
    model_file = Path(model_file)
    with np.load(model_file, allow_pickle=False) as archive:
        required = {
            "format_version",
            "model_kind",
            "feature_extractor",
            "labels",
            "weights",
            "bias",
            "feature_mean",
            "feature_scale",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"SVM model is missing arrays: {sorted(missing)}")
        version = int(np.asarray(archive["format_version"]).item())
        model_kind = str(np.asarray(archive["model_kind"]).item())
        extractor = str(np.asarray(archive["feature_extractor"]).item())
        labels = np.asarray(archive["labels"], dtype=np.int32)
        weights = np.asarray(archive["weights"], dtype=np.float32)
        bias = np.asarray(archive["bias"], dtype=np.float32)
        feature_mean = np.asarray(archive["feature_mean"], dtype=np.float32)
        feature_scale = np.asarray(archive["feature_scale"], dtype=np.float32)

    if version != 1 or model_kind != "linear_svm":
        raise ValueError(f"Unsupported SVM model format in: {model_file}")
    if extractor != expected_feature_extractor:
        raise ValueError(
            f"Model uses {extractor}, expected {expected_feature_extractor}"
        )
    if weights.ndim != 2 or labels.shape != (weights.shape[0],):
        raise ValueError("SVM labels and weights have incompatible shapes")
    if bias.shape != (weights.shape[0],):
        raise ValueError("SVM bias and weights have incompatible shapes")
    if feature_mean.shape != (1, weights.shape[1]) or feature_scale.shape != (
        1,
        weights.shape[1],
    ):
        raise ValueError("SVM normalization arrays have incompatible shapes")
    if not all(
        np.isfinite(array).all()
        for array in (weights, bias, feature_mean, feature_scale)
    ) or np.any(feature_scale <= 0.0):
        raise ValueError("SVM model contains invalid numeric values")
    return LinearSVM(labels, weights, bias, feature_mean, feature_scale)


def train_linear_svm(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 120,
    learning_rate: float = 0.02,
    regularization: float = 1e-4,
    seed: int = 2513,
) -> LinearSVM:
    """Train a deterministic multiclass linear SVM without extra dependencies."""
    if epochs < 1 or learning_rate <= 0.0 or regularization < 0.0:
        raise ValueError("Invalid SVM training parameter")

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Features and labels have incompatible shapes")

    feature_mean = features.mean(axis=0, keepdims=True).astype(np.float32)
    feature_scale = features.std(axis=0, keepdims=True).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    standardized = (features - feature_mean) / feature_scale

    unique_labels = np.unique(labels).astype(np.int32)
    weights = np.zeros((unique_labels.size, features.shape[1]), dtype=np.float32)
    bias = np.zeros(unique_labels.size, dtype=np.float32)
    binary_targets = np.where(
        labels[:, None] == unique_labels[None, :],
        1.0,
        -1.0,
    ).astype(np.float32)

    rng = np.random.default_rng(seed)
    sample_count = features.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(sample_count)
        epoch_features = standardized[order]
        epoch_targets = binary_targets[order]
        scores = epoch_features @ weights.T + bias
        active_targets = (
            ((1.0 - epoch_targets * scores) > 0.0).astype(np.float32)
            * epoch_targets
        )
        weight_gradient = (
            regularization * weights
            - (active_targets.T @ epoch_features) / sample_count
        )
        bias_gradient = -active_targets.mean(axis=0)
        step = learning_rate / math.sqrt(epoch + 1.0)
        weights -= step * weight_gradient
        bias -= step * bias_gradient

    return LinearSVM(
        labels=unique_labels,
        weights=weights,
        bias=bias,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def recognition_report(expected: np.ndarray, predicted: np.ndarray) -> dict:
    """Return overall and per-class recognition rates."""
    expected = np.asarray(expected, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)
    per_class: dict[str, dict[str, float | int]] = {}
    for label in np.unique(expected):
        selected = expected == label
        support = int(np.count_nonzero(selected))
        correct = int(np.count_nonzero(predicted[selected] == label))
        per_class[sign_description(label)] = {
            "support": support,
            "correct": correct,
            "recognition_rate": correct / support,
        }
    return {
        "sample_count": int(expected.size),
        "correct": int(np.count_nonzero(expected == predicted)),
        "recognition_rate": float(np.mean(expected == predicted)),
        "per_class": per_class,
    }


def extract_segmented_result_features(image_path: str | Path) -> np.ndarray | None:
    """Extract the project features from an already-segmented result image."""
    segmented = cv2.imread(str(image_path))
    if segmented is None:
        return None

    foreground_mask = np.any(segmented != 0, axis=2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        foreground_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if width == 0 or height == 0:
        return None

    contour_mask = np.zeros_like(foreground_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
    prepared_sign = cv2.resize(
        segmented[y : y + height, x : x + width],
        feature_extraction.IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    prepared_mask = cv2.resize(
        contour_mask[y : y + height, x : x + width],
        feature_extraction.IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    return np.concatenate(
        (
            feature_extraction._extract_hog_features(prepared_sign),
            feature_extraction._extract_hsv_features(prepared_sign, prepared_mask),
            feature_extraction._extract_shape_features(prepared_mask),
        )
    ).astype(np.float32)


def evaluate_result_folders(
    model: LinearSVM,
    result_directory: str | Path = DEFAULT_RESULT_DIRECTORY,
    expected_image_count: int = 84,
) -> dict:
    """Evaluate the trained model on the three labelled ColorInputs folders."""
    result_directory = Path(result_directory)
    feature_rows: list[np.ndarray] = []
    expected_labels: list[int] = []
    valid_groups: list[str] = []
    failed_images: list[str] = []
    group_totals: dict[str, int] = {}

    for group_name in RESULT_GROUPS:
        image_files = sorted(result_directory.joinpath(group_name).glob("*.png"))
        group_totals[group_name] = len(image_files)
        for image_path in image_files:
            features = feature_extraction.extract_features(image_path)
            if features is None:
                failed_images.append(str(image_path))
                continue
            feature_rows.append(features)
            expected_labels.append(sign_label_from_path(str(image_path)))
            valid_groups.append(group_name)

    total_images = sum(group_totals.values())
    if total_images != expected_image_count:
        raise ValueError(
            f"Expected {expected_image_count} result images, found {total_images}"
        )
    if not feature_rows:
        raise ValueError("No result images produced valid feature vectors")

    matrix = np.vstack(feature_rows).astype(np.float32)
    expected = np.asarray(expected_labels, dtype=np.int32)
    prediction_started = time.perf_counter()
    predicted = model.predict(matrix)
    prediction_seconds = time.perf_counter() - prediction_started
    correct_mask = predicted == expected
    group_array = np.asarray(valid_groups)
    per_group: dict[str, dict[str, float | int]] = {}
    for group_name in RESULT_GROUPS:
        selected = group_array == group_name
        correct = int(np.count_nonzero(correct_mask[selected]))
        total = group_totals[group_name]
        per_group[group_name] = {
            "total": total,
            "feature_ready": int(np.count_nonzero(selected)),
            "correct": correct,
            "recognition_rate": correct / total if total else 0.0,
        }

    correct = int(np.count_nonzero(correct_mask))
    return {
        "total_image_count": total_images,
        "feature_ready_count": int(expected.size),
        "failed_feature_count": len(failed_images),
        "failed_images": failed_images,
        "correct": correct,
        "recognition_rate": correct / total_images,
        "average_prediction_ms": prediction_seconds * 1000.0 / expected.size,
        "per_color": per_group,
        "classifier_ready": recognition_report(expected, predicted),
    }


def run_experiment(
    feature_file: str | Path = DEFAULT_FEATURE_FILE,
    validation_ratio: float = 0.2,
    seed: int = 2513,
    epochs: int = 120,
    samples_per_class: int = 160,
    result_directory: str | Path = DEFAULT_RESULT_DIRECTORY,
) -> tuple[LinearSVM, dict]:
    """Train the SVM and produce reproducible training and validation results."""
    features, labels = load_feature_dataset(feature_file)
    full_features, full_labels = features, labels
    # Remove exact duplicates before splitting; they remain valid training
    # samples when fitting the final model exclusively on TSRD.
    features, first, inverse = np.unique(
        features, axis=0, return_index=True, return_inverse=True
    )
    unique_labels = labels[first]
    if not np.array_equal(labels, unique_labels[inverse]):
        raise ValueError("Identical feature vectors have conflicting sign labels")
    labels = unique_labels
    train_indices, validation_indices, singleton_labels = stratified_holdout(
        labels,
        validation_ratio,
        seed,
    )
    training_features, training_labels = balance_training_classes(
        features[train_indices],
        labels[train_indices],
        samples_per_class=samples_per_class,
        seed=seed,
    )

    started = time.perf_counter()
    model = train_linear_svm(
        training_features,
        training_labels,
        epochs=epochs,
        seed=seed,
    )
    training_seconds = time.perf_counter() - started

    train_predictions = model.predict(features[train_indices])
    prediction_started = time.perf_counter()
    validation_predictions = model.predict(features[validation_indices])
    validation_seconds = time.perf_counter() - prediction_started

    final_training_features, final_training_labels = balance_training_classes(
        full_features,
        full_labels,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    final_training_started = time.perf_counter()
    final_model = train_linear_svm(
        final_training_features,
        final_training_labels,
        epochs=epochs,
        seed=seed,
    )
    final_training_seconds = time.perf_counter() - final_training_started

    report = {
        "model": "HOG HSV shape features with one-vs-rest linear SVM",
        "label_mapping": "three-digit filename prefix + 1 (000 -> Sign 1)",
        "feature_file": str(Path(feature_file).resolve()),
        "feature_length": int(features.shape[1]),
        "class_count": int(np.unique(labels).size),
        "unique_source_samples": int(features.shape[0]),
        "validation_policy": "Exact duplicate feature vectors removed before split",
        "seed": seed,
        "class_balancing": {
            "samples_per_class": samples_per_class,
            "original_training_samples": int(train_indices.size),
            "balanced_training_samples": int(training_labels.size),
        },
        "training_seconds": training_seconds,
        "average_validation_prediction_ms": (
            validation_seconds * 1000.0 / validation_indices.size
            if validation_indices.size
            else None
        ),
        "singleton_training_only_classes": [
            f"Sign {label}" for label in singleton_labels
        ],
        "training": recognition_report(
            labels[train_indices],
            train_predictions,
        ),
        "validation": recognition_report(
            labels[validation_indices],
            validation_predictions,
        ),
        "final_training": {
            "source_samples": int(full_features.shape[0]),
            "balanced_samples": int(final_training_labels.size),
            "training_seconds": final_training_seconds,
        },
        "result_validation": evaluate_result_folders(
            final_model,
            result_directory=result_directory,
        ),
    }
    return final_model, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_display_arguments(parser)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2513)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--samples-per-class", type=int, default=160)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="train again and replace the saved model",
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=DEFAULT_RESULT_DIRECTORY,
    )
    args = parser.parse_args()

    if args.model.is_file() and not args.retrain:
        model = load_linear_svm(args.model)
        report = {
            "model": "HOG HSV shape features with one-vs-rest linear SVM",
            "model_source": "loaded",
            "model_file": str(args.model.resolve()),
            "result_validation": evaluate_result_folders(
                model,
                result_directory=args.result_directory,
            ),
        }
    else:
        model, report = run_experiment(
            args.features,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            epochs=args.epochs,
            samples_per_class=args.samples_per_class,
            result_directory=args.result_directory,
        )
        save_linear_svm(model, args.model)
        report["model_source"] = "trained_and_saved"
        report["model_file"] = str(args.model.resolve())
    print(json.dumps(report, indent=2))
    correct_percentage = report["result_validation"]["recognition_rate"] * 100.0
    print(f"Correct percentage: {correct_percentage:.2f}%")
    display_predictions(model, feature_extraction, args)


if __name__ == "__main__":
    main()
