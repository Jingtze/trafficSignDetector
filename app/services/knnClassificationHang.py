"""k-nearest-neighbour classification using Hang's 34-value features.

This is the Hang-feature counterpart of ``knnClassification.py``.  It uses
the same filtered TSRD images, filename-to-sign mapping, reproducible split,
class balancing, and 84-image ColorInputs evaluation as the Hang SVM module.

Run from the repository root:
    python -m app.services.knnClassificationHang
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services import svmClassification as classification_base
from app.services import svmClassificationHang as hang_base


DEFAULT_MODEL_FILE = (
    Path(__file__).resolve().parent.parent / "models" / "knn_hang_model.npz"
)


@dataclass
class HangKNearestNeighbor:
    """Standardized Euclidean k-NN for Hang feature vectors."""

    training_features: np.ndarray
    training_labels: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    neighbors: int = 1

    def predict(self, feature_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(feature_rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != hang_base.HANG_FEATURE_VECTOR_LENGTH:
            raise ValueError(
                f"Expected feature rows of length "
                f"{hang_base.HANG_FEATURE_VECTOR_LENGTH}, received {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("Hang feature rows contain NaN or infinity")

        standardized = ((rows - self.feature_mean) / self.feature_scale).astype(
            np.float32
        )
        neighbor_count = min(self.neighbors, self.training_labels.size)
        training_squared = np.sum(
            self.training_features * self.training_features,
            axis=1,
        )
        predictions: list[int] = []

        for batch_start in range(0, standardized.shape[0], 128):
            batch = standardized[batch_start : batch_start + 128]
            distances = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + training_squared[None, :]
                - 2.0 * (batch @ self.training_features.T)
            )
            np.maximum(distances, 0.0, out=distances)
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
    model: HangKNearestNeighbor,
    model_file: str | Path = DEFAULT_MODEL_FILE,
) -> Path:
    """Save a trained Hang-feature k-NN model without using pickle."""
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_file,
        format_version=np.asarray(1, dtype=np.int32),
        model_kind=np.asarray("knn"),
        feature_extractor=np.asarray("featureExtractionHang.py"),
        training_features=np.asarray(model.training_features, dtype=np.float32),
        training_labels=np.asarray(model.training_labels, dtype=np.int32),
        feature_mean=np.asarray(model.feature_mean, dtype=np.float32),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float32),
        neighbors=np.asarray(model.neighbors, dtype=np.int32),
    )
    return model_file


def load_knn(model_file: str | Path = DEFAULT_MODEL_FILE) -> HangKNearestNeighbor:
    """Load and validate a Hang-feature k-NN model."""
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
            raise ValueError(f"Hang k-NN model is missing arrays: {sorted(missing)}")
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
        raise ValueError(f"Unsupported Hang k-NN model format in: {model_file}")
    if extractor != "featureExtractionHang.py":
        raise ValueError(
            f"Model uses {extractor}, expected featureExtractionHang.py"
        )
    expected_shape = (
        training_features.shape[0],
        hang_base.HANG_FEATURE_VECTOR_LENGTH,
    )
    if training_features.shape != expected_shape or training_labels.shape != (
        training_features.shape[0],
    ):
        raise ValueError("Hang k-NN training arrays have incompatible shapes")
    if feature_mean.shape != (1, hang_base.HANG_FEATURE_VECTOR_LENGTH) or (
        feature_scale.shape != (1, hang_base.HANG_FEATURE_VECTOR_LENGTH)
    ):
        raise ValueError("Hang k-NN normalization arrays have incompatible shapes")
    if neighbors < 1 or not all(
        np.isfinite(array).all()
        for array in (training_features, feature_mean, feature_scale)
    ) or np.any(feature_scale <= 0.0):
        raise ValueError("Hang k-NN model contains invalid values")
    return HangKNearestNeighbor(
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
) -> HangKNearestNeighbor:
    """Standardize and store Hang training rows for Euclidean k-NN."""
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    rows = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if (
        rows.ndim != 2
        or rows.shape[0] != labels.size
        or rows.shape[1] != hang_base.HANG_FEATURE_VECTOR_LENGTH
    ):
        raise ValueError("Hang features and labels have incompatible shapes")

    mean = rows.mean(axis=0, keepdims=True).astype(np.float32)
    scale = rows.std(axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((rows - mean) / scale).astype(np.float32)
    return HangKNearestNeighbor(
        training_features=standardized,
        training_labels=labels,
        feature_mean=mean,
        feature_scale=scale,
        neighbors=neighbors,
    )


def run_experiment(
    training_directory: str | Path = hang_base.DEFAULT_TRAINING_DIRECTORY,
    validation_ratio: float = 0.2,
    seed: int = 2513,
    neighbors: int = 1,
    samples_per_class: int = 160,
    result_directory: str | Path = hang_base.DEFAULT_RESULT_DIRECTORY,
) -> tuple[HangKNearestNeighbor, dict]:
    """Train and evaluate the reproducible Hang-feature k-NN model."""
    extracted, extracted_labels, failures = hang_base.load_hang_feature_dataset(
        training_directory
    )
    features, labels = hang_base.remove_exact_duplicates(
        extracted,
        extracted_labels,
    )
    train_indices, validation_indices, singleton_labels = (
        classification_base.stratified_holdout(labels, validation_ratio, seed)
    )
    training_features, training_labels = (
        classification_base.balance_training_classes(
            features[train_indices],
            labels[train_indices],
            samples_per_class=samples_per_class,
            seed=seed,
        )
    )

    started = time.perf_counter()
    model = train_knn(training_features, training_labels, neighbors=neighbors)
    training_seconds = time.perf_counter() - started
    training_predictions = model.predict(features[train_indices])
    validation_started = time.perf_counter()
    validation_predictions = model.predict(features[validation_indices])
    validation_seconds = time.perf_counter() - validation_started

    final_features, final_labels = classification_base.balance_training_classes(
        features,
        labels,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    final_started = time.perf_counter()
    final_model = train_knn(final_features, final_labels, neighbors=neighbors)
    final_training_seconds = time.perf_counter() - final_started

    report = {
        "model": f"Standardized Hang colour/shape/symbol features with {neighbors}-NN",
        "feature_extractor": "featureExtractionHang.py",
        "feature_components": {
            "colour_histogram": 20,
            "outer_shape_hu_moments": 7,
            "inner_symbol_hu_moments": 7,
        },
        "feature_length": hang_base.HANG_FEATURE_VECTOR_LENGTH,
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
            "neighbors": neighbors,
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
        "training": classification_base.recognition_report(
            labels[train_indices], training_predictions
        ),
        "validation": classification_base.recognition_report(
            labels[validation_indices], validation_predictions
        ),
        "final_training": {
            "balanced_samples": int(final_labels.size),
            "training_seconds": final_training_seconds,
        },
        "result_validation": hang_base.evaluate_result_folders(
            final_model,
            result_directory=result_directory,
        ),
    }
    return final_model, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-directory",
        type=Path,
        default=hang_base.DEFAULT_TRAINING_DIRECTORY,
    )
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
        default=hang_base.DEFAULT_RESULT_DIRECTORY,
    )
    args = parser.parse_args()

    if args.model.is_file() and not args.retrain:
        model = load_knn(args.model)
        report = {
            "model": (
                "Standardized Hang colour/shape/symbol features with "
                f"{model.neighbors}-NN"
            ),
            "model_source": "loaded",
            "model_file": str(args.model.resolve()),
            "result_validation": hang_base.evaluate_result_folders(
                model,
                result_directory=args.result_directory,
            ),
        }
    else:
        model, report = run_experiment(
            training_directory=args.training_directory,
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
    percentage = report["result_validation"]["recognition_rate"] * 100.0
    print(f"Correct percentage: {percentage:.2f}%")


if __name__ == "__main__":
    main()
