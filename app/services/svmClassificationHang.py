"""Linear SVM classification using ``featureExtractionHang.py`` features.

This is the Hang-feature counterpart of ``svmClassification.py``.  It builds
34-value colour, outer-shape, and inner-symbol feature vectors directly from
the filtered TSRD training images, then evaluates the fixed segmented images
in ``app/result/BlueSigns``, ``RedSigns``, and ``YellowSigns``.

Run from the repository root:
    python -m app.services.svmClassificationHang
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.services import featureExtractionHang as hang_features
from app.services import svmClassification as svm_base


DEFAULT_TRAINING_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "image"
    / "TSRDTrainingDataset"
    / "usable_training_images"
)
DEFAULT_RESULT_DIRECTORY = Path(__file__).resolve().parent.parent / "result"
DEFAULT_MODEL_FILE = (
    Path(__file__).resolve().parent.parent / "models" / "svm_hang_model.npz"
)
RESULT_GROUPS = ("BlueSigns", "RedSigns", "YellowSigns")
HANG_FEATURE_VECTOR_LENGTH = 34


def extract_hang_features_from_segmented_image(
    segmented_image: np.ndarray,
) -> np.ndarray:
    """Return Hang's 34-value feature vector for an already-segmented sign."""
    if segmented_image is None or segmented_image.size == 0:
        raise ValueError("Segmented image is empty")

    sign_image, sign_mask = hang_features.prepare_sign(segmented_image)
    vector = np.concatenate(
        (
            hang_features.collect_colour_features(sign_image, sign_mask),
            hang_features.collect_shape_features(sign_mask),
            hang_features.collect_symbol_features(sign_image, sign_mask),
        )
    ).astype(np.float32)
    if vector.shape != (HANG_FEATURE_VECTOR_LENGTH,):
        raise ValueError(
            f"Expected {HANG_FEATURE_VECTOR_LENGTH} Hang features, "
            f"received shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Hang feature vector contains NaN or infinity")
    return vector


def load_hang_feature_dataset(
    training_directory: str | Path = DEFAULT_TRAINING_DIRECTORY,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Extract Hang features and filename-derived labels from training PNGs."""
    training_directory = Path(training_directory)
    image_paths = sorted(training_directory.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG training images found in: {training_directory}")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    failures: list[dict[str, str]] = []
    for image_path in image_paths:
        try:
            row = np.asarray(hang_features.extract_features(image_path), dtype=np.float32)
            if row.shape != (HANG_FEATURE_VECTOR_LENGTH,):
                raise ValueError(
                    f"Expected {HANG_FEATURE_VECTOR_LENGTH} values, got {row.shape}"
                )
            if not np.isfinite(row).all():
                raise ValueError("Feature vector contains NaN or infinity")
            label = svm_base.sign_label_from_path(str(image_path))
        except (ValueError, cv2.error) as error:
            failures.append({"image": str(image_path), "error": str(error)})
            continue
        rows.append(row)
        labels.append(label)

    if not rows:
        raise ValueError("No training image produced a valid Hang feature vector")
    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        failures,
    )


def remove_exact_duplicates(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove identical feature rows so duplicates cannot cross the split."""
    unique, first, inverse = np.unique(
        features,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    unique_labels = labels[first]
    if not np.array_equal(labels, unique_labels[inverse]):
        raise ValueError("Identical Hang feature vectors have conflicting labels")
    return unique.astype(np.float32), unique_labels.astype(np.int32)


def extract_segmented_result_features(image_path: str | Path) -> np.ndarray | None:
    """Read one segmented result PNG and extract Hang features from it."""
    segmented = cv2.imread(str(image_path))
    if segmented is None:
        return None
    try:
        return extract_hang_features_from_segmented_image(segmented)
    except (ValueError, cv2.error):
        return None


def evaluate_result_folders(
    model: object,
    result_directory: str | Path = DEFAULT_RESULT_DIRECTORY,
    expected_image_count: int = 84,
) -> dict:
    """Evaluate a Hang-feature classifier on the fixed segmented results."""
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
            vector = extract_segmented_result_features(image_path)
            if vector is None:
                failed_images.append(str(image_path))
                continue
            feature_rows.append(vector)
            expected_labels.append(svm_base.sign_label_from_path(str(image_path)))
            valid_groups.append(group_name)

    total_images = sum(group_totals.values())
    if total_images != expected_image_count:
        raise ValueError(
            f"Expected {expected_image_count} result images, found {total_images}"
        )
    if not feature_rows:
        raise ValueError("No result image produced a valid Hang feature vector")

    matrix = np.vstack(feature_rows).astype(np.float32)
    expected = np.asarray(expected_labels, dtype=np.int32)
    prediction_started = time.perf_counter()
    predicted = np.asarray(model.predict(matrix), dtype=np.int32)
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
        "classifier_ready": svm_base.recognition_report(expected, predicted),
    }


def run_experiment(
    training_directory: str | Path = DEFAULT_TRAINING_DIRECTORY,
    validation_ratio: float = 0.2,
    seed: int = 2513,
    epochs: int = 120,
    learning_rate: float = 0.02,
    regularization: float = 1e-4,
    samples_per_class: int = 160,
    result_directory: str | Path = DEFAULT_RESULT_DIRECTORY,
) -> tuple[svm_base.LinearSVM, dict]:
    """Train and evaluate the reproducible Hang-feature linear SVM."""
    extracted, extracted_labels, failures = load_hang_feature_dataset(
        training_directory
    )
    features, labels = remove_exact_duplicates(extracted, extracted_labels)
    train_indices, validation_indices, singleton_labels = svm_base.stratified_holdout(
        labels,
        validation_ratio,
        seed,
    )
    training_features, training_labels = svm_base.balance_training_classes(
        features[train_indices],
        labels[train_indices],
        samples_per_class=samples_per_class,
        seed=seed,
    )

    started = time.perf_counter()
    model = svm_base.train_linear_svm(
        training_features,
        training_labels,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=seed,
    )
    training_seconds = time.perf_counter() - started
    training_predictions = model.predict(features[train_indices])
    validation_started = time.perf_counter()
    validation_predictions = model.predict(features[validation_indices])
    validation_seconds = time.perf_counter() - validation_started

    final_features, final_labels = svm_base.balance_training_classes(
        features,
        labels,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    final_started = time.perf_counter()
    final_model = svm_base.train_linear_svm(
        final_features,
        final_labels,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=seed,
    )
    final_training_seconds = time.perf_counter() - final_started

    report = {
        "model": "Hang colour/shape/symbol features with one-vs-rest linear SVM",
        "feature_extractor": "featureExtractionHang.py",
        "feature_components": {
            "colour_histogram": 20,
            "outer_shape_hu_moments": 7,
            "inner_symbol_hu_moments": 7,
        },
        "feature_length": HANG_FEATURE_VECTOR_LENGTH,
        "label_mapping": "three-digit filename prefix + 1 (000 -> Sign 1)",
        "training_directory": str(Path(training_directory).resolve()),
        "source_image_count": int(extracted.shape[0] + len(failures)),
        "successful_source_images": int(extracted.shape[0]),
        "failed_source_images": len(failures),
        "training_failures": failures,
        "unique_source_samples": int(features.shape[0]),
        "class_count": int(np.unique(labels).size),
        "validation_policy": "Exact duplicate Hang vectors removed before split",
        "seed": seed,
        "parameters": {
            "validation_ratio": validation_ratio,
            "samples_per_class": samples_per_class,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "regularization": regularization,
        },
        "class_balancing": {
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
        "training": svm_base.recognition_report(
            labels[train_indices], training_predictions
        ),
        "validation": svm_base.recognition_report(
            labels[validation_indices], validation_predictions
        ),
        "final_training": {
            "balanced_samples": int(final_labels.size),
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
    parser.add_argument(
        "--training-directory", type=Path, default=DEFAULT_TRAINING_DIRECTORY
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2513)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--samples-per-class", type=int, default=160)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="train again and replace the saved model",
    )
    parser.add_argument(
        "--result-directory", type=Path, default=DEFAULT_RESULT_DIRECTORY
    )
    args = parser.parse_args()

    if args.model.is_file() and not args.retrain:
        model = svm_base.load_linear_svm(
            args.model,
            expected_feature_extractor="featureExtractionHang.py",
        )
        if model.weights.shape[1] != HANG_FEATURE_VECTOR_LENGTH:
            raise ValueError(
                f"Hang SVM must contain {HANG_FEATURE_VECTOR_LENGTH} features"
            )
        report = {
            "model": (
                "Hang colour/shape/symbol features with one-vs-rest linear SVM"
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
            training_directory=args.training_directory,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            samples_per_class=args.samples_per_class,
            result_directory=args.result_directory,
        )
        svm_base.save_linear_svm(
            model,
            args.model,
            feature_extractor="featureExtractionHang.py",
        )
        report["model_source"] = "trained_and_saved"
        report["model_file"] = str(args.model.resolve())
    print(json.dumps(report, indent=2))
    percentage = report["result_validation"]["recognition_rate"] * 100.0
    print(f"Correct percentage: {percentage:.2f}%")


if __name__ == "__main__":
    main()
