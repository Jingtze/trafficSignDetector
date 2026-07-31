import cv2
import numpy as np
from pathlib import Path


def segment_red_color(image_path):
    # image receive
    before_gau = cv2.imread(str(image_path))

    # preprocessing
    if before_gau is None:
        print(f"Image Not Found: {image_path}")
        return None

    img_area = before_gau.shape[0] * before_gau.shape[1]
    image = cv2.GaussianBlur(before_gau, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 85, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 85, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)

    water_red_mask = watershed_segmentation(image, red_mask, 3, 0.5)
    cv2.imshow("water",water_red_mask)
    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(red_mask)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        shape = shape_detection(contour, img_area)
        if shape is None:
            continue
        elif shape == "recover":
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            print(aspect_ratio)
            if 0.75 <= aspect_ratio <= 1.1:
                print(1)
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(filled_mask, ellipse, 255, thickness=-1)

        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)
    cv2.imshow("segment", segmented_sign)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def segment_blue_color(image_path):
    # image receive
    before_gau = cv2.imread(str(image_path))

    # preprocessing
    if before_gau is None:
        print(f"Image Not Found: {image_path}")
        return None

    img_area = before_gau.shape[0] * before_gau.shape[1]
    image = cv2.GaussianBlur(before_gau, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    v_clahe = clahe.apply(v)
    hsv = cv2.merge([h, s, v_clahe])

    lower_blue = np.array([95, 100, 40])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_open)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)

    water_blue_mask = watershed_segmentation(image, blue_mask, 3, 0.55)

    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(water_blue_mask)
    contours, _ = cv2.findContours(water_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        shape = shape_detection(contour, img_area)
        if shape is None:
            continue

        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)
    cv2.imshow("segment", segmented_sign)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segmented_sign


def segment_yellow_color(image_path):
    # image receive
    before_gau = cv2.imread(str(image_path))

    # preprocessing
    if before_gau is None:
        print(f"Image Not Found: {image_path}")
        return

    img_area = before_gau.shape[0] * before_gau.shape[1]
    image = cv2.GaussianBlur(before_gau, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Prepare an edge image for the yellow shape fallback.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel)

    cv2.imshow("0. Original Yellow Image", before_gau)
    cv2.imshow("1. Yellow Sign Edges", closed_edges)

    lower_yellow = np.array([15, 90, 50])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    raw_yellow_mask = yellow_mask.copy()

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_open)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_close)

    water_yellow_mask = watershed_segmentation(image, yellow_mask, 3, 0.50)
    cv2.imshow("wateryellow", water_yellow_mask)

    mask = water_yellow_mask.copy()

    h, w = mask.shape[:2]

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )
    cv2.floodFill(
        mask,
        flood_mask,
        (0, 0),
        255
    )
    flood_inverse = cv2.bitwise_not(mask)
    yellow_mask = water_yellow_mask | flood_inverse
    cv2.imshow("flood", yellow_mask)

    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(yellow_mask)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sign_found = False

    for contour in contours:

        shape = shape_detection(contour, img_area)
        if shape is None:
            continue

        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

        if shape in ("triangle", "rectangle"):
            sign_found = True

    edge_shape = find_yellow_edge_shape(
        closed_edges,
        raw_yellow_mask,
        img_area
    )
    edge_preview = before_gau.copy()

    if edge_shape is not None:
        edge_shape_mask = np.zeros_like(yellow_mask)
        cv2.drawContours(
            edge_shape_mask,
            [edge_shape],
            0,
            255,
            thickness=-1
        )

        # A one-pixel expansion helps include the thin outer border.
        if len(edge_shape) == 3:
            expand_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
            edge_shape_mask = cv2.dilate(
                edge_shape_mask,
                expand_kernel,
                iterations=2
            )

        # Prefer a verified triangle. Use a rectangle only as a true fallback.
        if len(edge_shape) == 3 or not sign_found:
            filled_mask = edge_shape_mask

        cv2.drawContours(
            edge_preview,
            [edge_shape],
            0,
            (0, 255, 0),
            thickness=2
        )

    cv2.imshow("2. Edge Shape Candidate", edge_preview)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)

    cv2.imshow("3. Segmented Sign", segmented_sign)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_yellow_edge_shape(edge_mask, raw_yellow_mask, img_area):
    contours, hierarchy = cv2.findContours(
        edge_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None:
        hierarchy = hierarchy[0]

    best_triangle = None
    best_triangle_score = -1
    best_triangle_area = 0
    best_rectangle = None
    best_rectangle_score = -1
    best_rectangle_area = 0
    image_height, image_width = raw_yellow_mask.shape[:2]

    for index, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = None
        for epsilon_ratio in (0.02, 0.03, 0.04):
            candidate = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True
            )
            if len(candidate) in (3, 4):
                approx = candidate
                break

        if approx is None:
            continue

        if not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < img_area * 0.01 or area > img_area * 0.75:
            continue

        x, y, width, height = cv2.boundingRect(approx)
        if width == 0 or height == 0:
            continue

        fill_ratio = area / float(width * height)

        if len(approx) == 3:
            points = approx.reshape(3, 2)
            top_index = np.argmin(points[:, 1])
            top_point = points[top_index]
            bottom_points = np.delete(points, top_index, axis=0)

            base_y_difference = (
                    abs(bottom_points[0][1] - bottom_points[1][1])
                    / height
            )
            base_width_ratio = (
                    abs(bottom_points[0][0] - bottom_points[1][0])
                    / height
            )
            bottom_min_x = min(
                bottom_points[0][0],
                bottom_points[1][0]
            )
            bottom_max_x = max(
                bottom_points[0][0],
                bottom_points[1][0]
            )

            # Reject flat or sideways triangles created by symbols and borders.
            if fill_ratio < 0.25:
                continue
            if base_y_difference > 0.30:
                continue
            if not bottom_min_x <= top_point[0] <= bottom_max_x:
                continue
            if not 0.50 <= base_width_ratio <= 2.50:
                continue

        else:
            aspect_ratio = width / float(height)

            # A sign board should be rectangular and should not extend into
            # the post or background at the bottom of the image.
            if fill_ratio < 0.55:
                continue
            if not 0.45 <= aspect_ratio <= 2.20:
                continue
            if y + height >= image_height - 1:
                continue
            if width >= image_width * 0.95:
                continue

        candidate_mask = np.zeros_like(raw_yellow_mask)
        cv2.drawContours(
            candidate_mask,
            [approx],
            0,
            255,
            thickness=-1
        )

        candidate_area = cv2.countNonZero(candidate_mask)
        if candidate_area == 0:
            continue

        yellow_inside = cv2.bitwise_and(
            raw_yellow_mask,
            candidate_mask
        )
        yellow_ratio = (
                cv2.countNonZero(yellow_inside) / candidate_area
        )

        if yellow_ratio < 0.03:
            continue

        area_ratio = area / img_area
        nested_bonus = 0
        if hierarchy is not None:
            has_child = hierarchy[index][2] != -1
            has_parent = hierarchy[index][3] != -1
            if has_child or has_parent:
                nested_bonus = 0.50

        score = (
                (0.25 * yellow_ratio)
                + (3.0 * area_ratio)
                + nested_bonus
        )

        if len(approx) == 3 and score > best_triangle_score:
            best_triangle = approx
            best_triangle_score = score
            best_triangle_area = area
        elif len(approx) == 4 and score > best_rectangle_score:
            best_rectangle = approx
            best_rectangle_score = score
            best_rectangle_area = area

    if best_triangle is not None and best_rectangle is not None:
        triangle_points = best_triangle.reshape(-1, 2)
        triangle_center = tuple(
            np.mean(triangle_points, axis=0).astype(float)
        )
        rectangle_contains_triangle = (
                cv2.pointPolygonTest(
                    best_rectangle,
                    triangle_center,
                    False
                )
                >= 0
        )

        # Keep a genuine sign board when it cleanly surrounds the triangle.
        if (
                rectangle_contains_triangle
                and best_rectangle_area >= best_triangle_area * 1.25
        ):
            return best_rectangle

    if best_triangle is not None:
        return best_triangle

    return best_rectangle


def shape_detection(contour, img_area):
    if len(contour) < 3:
        return None

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter == 0:
        return None

    min_area = img_area * 0.05
    if area < min_area:
        return None

    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    circle_ratio = area / circle_area if circle_area > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area)
    if solidity < 0.75:
        return "recover"

    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        return "rectangle"
    elif circle_ratio > 0.80:

        return "circle"
    elif 7 <= vertices <= 9:
        return "octagon"
    else:
        return None


def watershed_segmentation(img, mask, iterations, therehold):
    if cv2.countNonZero(mask) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=iterations)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, therehold * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    result = np.zeros_like(mask)
    result[markers > 1] = 255

    return result


BASE_DIR = Path(__file__).resolve().parent.parent
image_path = BASE_DIR / "image" / "ColorInputs"
signs = ["RedSigns", "BlueSigns", "YellowSigns"]
image_files = list((image_path / signs[0]).glob("*.png"))

for item in image_files:
    segment_red_color(item)


