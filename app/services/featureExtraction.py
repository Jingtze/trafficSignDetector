"""Feature extraction for segmented traffic-sign images."""

import cv2
import numpy as np

from app.services.opencvOperation import segment_traffic_sign


__all__ = ["extract_features", "extract_feature_matrix", "FEATURE_VECTOR_LENGTH"]


IMAGE_SIZE = (64, 64)
HOG_FEATURE_LENGTH = 7 * 7 * 2 * 2 * 9
FEATURE_VECTOR_LENGTH = HOG_FEATURE_LENGTH + (18 * 8) + 13


def _prepare_segmented_sign(image_path):
    """Segment, crop, and resize one traffic sign."""
    segmented = segment_traffic_sign(image_path)
    if segmented is None:
        return None, None

    foreground_mask = np.any(segmented != 0, axis=2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        foreground_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)
    if width == 0 or height == 0:
        return None, None

    contour_mask = np.zeros_like(foreground_mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=-1)

    cropped_sign = segmented[y : y + height, x : x + width]
    cropped_mask = contour_mask[y : y + height, x : x + width]

    prepared_sign = cv2.resize(cropped_sign, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    prepared_mask = cv2.resize(
        cropped_mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    return prepared_sign, prepared_mask


def _extract_hog_features(prepared_sign):
    grayscale = cv2.cvtColor(prepared_sign, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(
        gradient_x,
        gradient_y,
        angleInDegrees=True,
    )

    angle = np.mod(angle, 180.0)
    orientation_bins = np.floor(angle / 20.0).astype(np.int32)
    cell_histograms = np.zeros((8, 8, 9), dtype=np.float32)

    for cell_y in range(8):
        for cell_x in range(8):
            y_start = cell_y * 8
            x_start = cell_x * 8
            cell_bins = orientation_bins[
                y_start : y_start + 8,
                x_start : x_start + 8,
            ].reshape(-1)
            cell_magnitudes = magnitude[
                y_start : y_start + 8,
                x_start : x_start + 8,
            ].reshape(-1)
            cell_histograms[cell_y, cell_x] = np.bincount(
                cell_bins,
                weights=cell_magnitudes,
                minlength=9,
            )

    normalized_blocks = []
    for block_y in range(7):
        for block_x in range(7):
            block = cell_histograms[
                block_y : block_y + 2,
                block_x : block_x + 2,
            ].reshape(-1)
            block /= np.sqrt(np.sum(block * block) + 1e-6)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.sum(block * block) + 1e-6)
            normalized_blocks.append(block)

    return np.concatenate(normalized_blocks).astype(np.float32)


def _extract_hsv_features(prepared_sign, sign_mask):
    hsv = cv2.cvtColor(prepared_sign, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        sign_mask,
        [18, 8],
        [0, 180, 0, 256],
    )
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram.reshape(-1)


def _extract_shape_features(sign_mask):
    contours, _ = cv2.findContours(
        sign_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, width, height = cv2.boundingRect(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    image_area = float(sign_mask.shape[0] * sign_mask.shape[1])

    area_ratio = area / image_area
    aspect_ratio = width / float(height) if height else 0.0
    extent = area / float(width * height) if width and height else 0.0
    solidity = area / hull_area if hull_area else 0.0
    circularity = (
        4.0 * np.pi * area / (perimeter * perimeter)
        if perimeter
        else 0.0
    )

    polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertex_count = min(len(polygon), 20) / 20.0

    hu_moments = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)

    basic_features = np.array(
        [
            area_ratio,
            aspect_ratio,
            extent,
            solidity,
            circularity,
            vertex_count,
        ],
        dtype=np.float32,
    )
    return np.concatenate((basic_features, hu_moments.astype(np.float32)))


def extract_features(image_path):
    """Return HOG, HSV, and shape features as one float32 NumPy array."""
    prepared_sign, sign_mask = _prepare_segmented_sign(image_path)
    if prepared_sign is None:
        return None

    hog_features = _extract_hog_features(prepared_sign)
    hsv_features = _extract_hsv_features(prepared_sign, sign_mask)
    shape_features = _extract_shape_features(sign_mask)

    return np.concatenate(
        (hog_features, hsv_features, shape_features)
    ).astype(np.float32)


def extract_feature_matrix(image_paths):
    """Return one classifier-ready row per image, preserving input order."""
    feature_rows = []
    for image_path in image_paths:
        feature_vector = extract_features(image_path)
        if feature_vector is None:
            raise ValueError(f"Could not extract features from: {image_path}")
        feature_rows.append(feature_vector)

    if not feature_rows:
        return np.empty((0, FEATURE_VECTOR_LENGTH), dtype=np.float32)

    return np.vstack(feature_rows).astype(np.float32)
