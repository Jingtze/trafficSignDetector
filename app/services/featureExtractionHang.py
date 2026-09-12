import cv2
import numpy as np


from app.services.opencvOperation import segment_traffic_sign

__all__ = ["extract_features"]

def prepare_sign(segmented_sign):

    visible_mask = np.any(segmented_sign !=0, axis=2)
    visible_mask = visible_mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No value found.")

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) <= 0:
        raise ValueError("The value is too small.")

    sign_mask = np.zeros(segmented_sign.shape[:2], dtype=np.uint8)
    cv2.drawContours(sign_mask, [contour], -1, 255, thickness=-1)

    x, y, width, height = cv2.boundingRect(contour)

    sign_image = segmented_sign[y:y + height, x:x + width]
    cropped_mask = sign_mask[y:y + height, x:x + width]

    return sign_image, cropped_mask

def collect_colour_features(sign_image, sign_mask):

    hsv = cv2.cvtColor(sign_image, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], sign_mask, [10], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], sign_mask, [10], [0, 256])

    pixel_count = cv2.countNonZero(sign_mask)

    if pixel_count == 0:
        raise ValueError("The sign mask is empty.")

    h_features = h_hist.flatten() / pixel_count
    s_features = s_hist.flatten() / pixel_count

    return np.concatenate((h_features, s_features))

def collect_shape_features(sign_mask):

    moments = cv2.moments(sign_mask, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()

    shape_features = np.zeros(7, dtype=np.float64)

    for i, value in enumerate(hu):
        if value != 0:
            shape_features[i] = (-np.sign(value) * np.log10(abs(value)))

    return shape_features

def collect_symbol(sign_image, sign_mask):

    turn_to_gray = cv2.cvtColor(sign_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sign_image, cv2.COLOR_BGR2HSV)

    height, width = sign_mask.shape

    border_size = max(1, round(min(height, width) * 0.10))

    inside = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_size * 2 + 1, border_size * 2 + 1)
    )

    inner_mask = cv2.erode(sign_mask, inside, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    inside_sign = inner_mask > 0
    pixels_inside = turn_to_gray[inner_mask > 0]

    if pixels_inside.size == 0:
        return np.zeros_like(sign_mask)

    if pixels_inside.min() == pixels_inside.max():
        return np.zeros_like(sign_mask)

    threshold, _ = cv2.threshold(pixels_inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    coloured_pixels = (saturation >= 80) & (value >= 40)

    red_pixels = (((hue <= 10) | (hue >= 170)) & coloured_pixels & inside_sign)
    blue_pixels = ((hue >= 95) & (hue <= 130) & coloured_pixels & inside_sign)

    inside_count = pixels_inside.size

    red_ratio = np.count_nonzero(red_pixels) / inside_count
    blue_ratio = np.count_nonzero(blue_pixels) / inside_count

    if red_ratio > 0.5 or blue_ratio > 0.5:
        symbol_pixels = turn_to_gray > threshold
    else:
        symbol_pixels = turn_to_gray <= threshold

    symbol_mask = np.zeros_like(sign_mask)

    symbol_mask[symbol_pixels & inside_sign] = 255

    return symbol_mask

def collect_symbol_features(sign_image, sign_mask):

    symbol_mask = collect_symbol(sign_image, sign_mask)

    if cv2.countNonZero(symbol_mask) == 0:
        raise ValueError("No inner symbol found.")

    return collect_shape_features(symbol_mask)


def extract_features(image_path):

    segmented_sign = segment_traffic_sign(image_path)

    if segmented_sign is None:
        raise ValueError(f"Segmentation failed: {image_path}")

    sign_image, sign_mask = prepare_sign(segmented_sign)

    color_features = collect_colour_features(sign_image, sign_mask)
    shape_features = collect_shape_features(sign_mask)
    symbol_features = collect_symbol_features(sign_image, sign_mask)

    features = np.concatenate((
        color_features,
        shape_features,
        symbol_features
    )).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError(f"Invalid feature values: {image_path}")

    return features




