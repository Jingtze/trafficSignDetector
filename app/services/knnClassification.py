"""k-nearest-neighbour classification for traffic-sign feature vectors.

This comparison model consumes the same overlap-free
``tsrd_training_features.npz`` output as the SVM classifier. It does not repeat
segmentation or feature extraction. The three-digit filename prefix is
converted to the confirmed one-based sign ID, so prefix 000 is reported as
Sign 1.

Run from the repository root:
    python -m app.services.knnClassification
"""

from __future__ import annotations

import argparse
import json
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
    Path(__file__).resolve().parent.parent / "models" / "knn_normal_model.npz"
)
RESULT_GROUPS = ("BlueSigns", "RedSigns", "YellowSigns")
LABEL_PATTERN = re.compile(r"(?:^|[\\/])(\d{3})(?:_|$)")


def sign_label_from_path(image_path: str) -> int:
    """Convert a stored filename prefix to the confirmed one-based sign ID."""
    match = LABEL_PATTERN.search(str(image_path))
    if match is None:
        raise ValueError(f"Cannot find a three-digit sign label in: {image_path}")
    return int(match.group(1)) + 1


def load_feature_dataset(feature_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load classifier-ready features and derive labels from image paths."""
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
    # Exact duplicates must not cross the internal train/validation boundary.
    unique, first, inverse = np.unique(
        features, axis=0, return_index=True, return_inverse=True
    )
    unique_labels = labels[first]
    if not np.array_equal(labels, unique_labels[inverse]):
        raise ValueError("Identical feature vectors have conflicting sign labels")
    return unique, unique_labels


def transform_classifier_features(features: np.ndarray) -> np.ndarray:
    """Normalize HOG and square-root HSV blocks before classifier training.

    Input remains the existing 1,921-value extractor output. Shape values are
    omitted after TSRD validation selection. Training and inference use the
    same fixed transform; HSV weight is 0.25 relative to normalized HOG.
    """
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 1921:
        raise ValueError("Expected finite feature rows of length 1921")
    if not np.isfinite(rows).all():
        raise ValueError("Feature rows contain NaN or infinity")
    hog = rows[:, :1764].copy()
    hog /= np.maximum(np.linalg.norm(hog, axis=1, keepdims=True), 1e-8)
    hsv = np.sqrt(np.maximum(rows[:, 1764:1908], 0.0))
    hsv /= np.maximum(np.linalg.norm(hsv, axis=1, keepdims=True), 1e-8)
    return np.concatenate((hog, 0.25 * hsv), axis=1).astype(np.float32)


def stratified_holdout(
    labels: np.ndarray,
    validation_ratio: float = 0.2,
    seed: int = 2513,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Split each repeated class while retaining singleton classes for training."""
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
class KNearestNeighbor:
    """Standardized Euclidean k-nearest-neighbour classifier."""

    training_features: np.ndarray
    training_labels: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    neighbors: int = 1

    def predict(self, feature_rows: np.ndarray) -> np.ndarray:
        rows = transform_classifier_features(feature_rows)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != self.training_features.shape[1]:
            raise ValueError(
                f"Expected feature rows of length {self.training_features.shape[1]}, "
                f"received shape {rows.shape}"
            )

        standardized = ((rows - self.feature_mean) / self.feature_scale).astype(
            np.float32
        )
        neighbor_count = min(self.neighbors, self.training_labels.size)
        predictions: list[int] = []
        training_squared = np.sum(
            self.training_features * self.training_features,
            axis=1,
        )
        batch_size = 128
        for batch_start in range(0, standardized.shape[0], batch_size):
            batch = standardized[batch_start : batch_start + batch_size]
            batch_squared = np.sum(batch * batch, axis=1, keepdims=True)
            distances = (
                batch_squared
                + training_squared[None, :]
                - 2.0 * (batch @ self.training_features.T)
            )
            np.maximum(distances, 0.0, out=distances)

            if neighbor_count == 1:
                nearest_indices = np.argmin(distances, axis=1)
                predictions.extend(
                    self.training_labels[nearest_indices].astype(int).tolist()
                )
                continue

            nearest_rows = np.argpartition(
                distances,
                neighbor_count - 1,
                axis=1,
            )[:, :neighbor_count]
            for row_number, nearest_indices in enumerate(nearest_rows):
                nearest_labels = self.training_labels[nearest_indices]
                nearest_distances = distances[row_number, nearest_indices]
                candidates: list[tuple[int, float, int]] = []
                for label in np.unique(nearest_labels):
                    selected = nearest_labels == label
                    candidates.append(
                        (
                            -int(np.count_nonzero(selected)),
                            float(np.sum(nearest_distances[selected])),
                            int(label),
                        )
                    )
                predictions.append(min(candidates)[2])
        return np.asarray(predictions, dtype=np.int32)

    def predict_sign(self, feature_vector: np.ndarray) -> str:
        return f"Sign {int(self.predict(feature_vector)[0])}"


def save_knn(
    model: KNearestNeighbor,
    model_file: str | Path = DEFAULT_MODEL_FILE,
) -> Path:
    """Save a trained normal-feature k-NN model without using pickle."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_file,
        format_version=np.asarray(1, dtype=np.int32),
        model_kind=np.asarray("knn"),
        feature_extractor=np.asarray("featureExtraction.py"),
        training_features=np.asarray(model.training_features, dtype=np.float32),
        training_labels=np.asarray(model.training_labels, dtype=np.int32),
        feature_mean=np.asarray(model.feature_mean, dtype=np.float32),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float32),
        neighbors=np.asarray(model.neighbors, dtype=np.int32),
    )
    return model_file


def load_knn(model_file: str | Path = DEFAULT_MODEL_FILE) -> KNearestNeighbor:
    """Load and validate a normal-feature k-NN model."""
    model_file = Path(model_file)
    with np.load(model_file, allow_pickle=False) as archive:
        required = {
            "format_version",
            "model_kind",
            "feature_extractor",
            "training_features",
            "training_labels",
            "feature_mean",
            "feature_scale",
            "neighbors",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"k-NN model is missing arrays: {sorted(missing)}")
        version = int(np.asarray(archive["format_version"]).item())
        model_kind = str(np.asarray(archive["model_kind"]).item())
        extractor = str(np.asarray(archive["feature_extractor"]).item())
        training_features = np.asarray(
            archive["training_features"], dtype=np.float32
        )
        training_labels = np.asarray(archive["training_labels"], dtype=np.int32)
        feature_mean = np.asarray(archive["feature_mean"], dtype=np.float32)
        feature_scale = np.asarray(archive["feature_scale"], dtype=np.float32)
        neighbors = int(np.asarray(archive["neighbors"]).item())

    if version != 1 or model_kind != "knn":
        raise ValueError(f"Unsupported k-NN model format in: {model_file}")
    if extractor != "featureExtraction.py":
        raise ValueError(f"Model uses {extractor}, expected featureExtraction.py")
    if training_features.ndim != 2 or training_labels.shape != (
        training_features.shape[0],
    ):
        raise ValueError("k-NN training arrays have incompatible shapes")
    feature_length = training_features.shape[1]
    if feature_mean.shape != (1, feature_length) or feature_scale.shape != (
        1,
        feature_length,
    ):
        raise ValueError("k-NN normalization arrays have incompatible shapes")
    if neighbors < 1 or not all(
        np.isfinite(array).all()
        for array in (training_features, feature_mean, feature_scale)
    ) or np.any(feature_scale <= 0.0):
        raise ValueError("k-NN model contains invalid values")
    return KNearestNeighbor(
        training_features,
        training_labels,
        feature_mean,
        feature_scale,
        neighbors,
    )


def train_knn(
    features: np.ndarray,
    labels: np.ndarray,
    neighbors: int = 1,
) -> KNearestNeighbor:
    """Store standardized training rows for k-NN inference."""
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("Features and labels have incompatible shapes")

    features = transform_classifier_features(features)
    feature_mean = np.zeros((1, features.shape[1]), dtype=np.float32)
    feature_scale = np.ones((1, features.shape[1]), dtype=np.float32)
    standardized = ((features - feature_mean) / feature_scale).astype(np.float32)
    return KNearestNeighbor(
        training_features=standardized,
        training_labels=labels,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        neighbors=neighbors,
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
        per_class[f"Sign {int(label)}"] = {
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
    model: KNearestNeighbor,
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
    neighbors: int = 1,
    samples_per_class: int = 160,
    result_directory: str | Path = DEFAULT_RESULT_DIRECTORY,
) -> tuple[KNearestNeighbor, dict]:
    """Train k-NN and produce reproducible training and validation results."""
    features, labels = load_feature_dataset(feature_file)
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
    model = train_knn(
        training_features,
        training_labels,
        neighbors=neighbors,
    )
    training_seconds = time.perf_counter() - started

    train_predictions = model.predict(features[train_indices])
    prediction_started = time.perf_counter()
    validation_predictions = model.predict(features[validation_indices])
    validation_seconds = time.perf_counter() - prediction_started

    final_training_features, final_training_labels = balance_training_classes(
        features,
        labels,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    final_training_started = time.perf_counter()
    final_model = train_knn(
        final_training_features,
        final_training_labels,
        neighbors=neighbors,
    )
    final_training_seconds = time.perf_counter() - final_training_started

    report = {
        "model": f"Normalized HOG and square-root HSV features with {neighbors}-NN",
        "label_mapping": "three-digit filename prefix + 1 (000 -> Sign 1)",
        "feature_file": str(Path(feature_file).resolve()),
        "feature_length": int(features.shape[1]),
        "class_count": int(np.unique(labels).size),
        "unique_source_samples": int(features.shape[0]),
        "validation_policy": "Exact duplicate feature vectors removed before split",
        "classifier_features": "L2 HOG + 0.25 * L2 square-root HSV; shape omitted",
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
            "source_samples": int(features.shape[0]),
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
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2513)
    parser.add_argument("--neighbors", type=int, default=1)
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
        model = load_knn(args.model)
        report = {
            "model": (
                "Normalized HOG and square-root HSV features with "
                f"{model.neighbors}-NN"
            ),
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
            neighbors=args.neighbors,
            samples_per_class=args.samples_per_class,
            result_directory=args.result_directory,
        )
        save_knn(model, args.model)
        report["model_source"] = "trained_and_saved"
        report["model_file"] = str(args.model.resolve())
    print(json.dumps(report, indent=2))
    correct_percentage = report["result_validation"]["recognition_rate"] * 100.0
    print(f"Correct percentage: {correct_percentage:.2f}%")


if __name__ == "__main__":
    main()
