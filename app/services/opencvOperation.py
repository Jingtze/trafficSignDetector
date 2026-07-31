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
    image = cv2.GaussianBlur(before_gau, (3,3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    lower1 = np.array([0, 75, 60])

    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 75, 60])

    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))


    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)

    filled_mask = np.zeros_like(red_mask)
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_found=False
    parent=None
    hierarchy=hierarchy[0]
    for i, contour in enumerate(contours):
        shape = shape_detection(contour, img_area)

        if shape == "recover":
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.75 <= aspect_ratio <= 1.1:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(filled_mask, ellipse, 255, thickness=-1)
                continue
        elif shape is not None:
            cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)
            continue

        outer_area = cv2.contourArea(contour)

        if outer_area < 500:
            continue

        child = hierarchy[i][2]

        # check if inner contour exists
        while child != -1:

            inner_contour = contours[child]

            inner_area = cv2.contourArea(inner_contour)

            ratio = inner_area / outer_area

            if ratio > 0.5:
                inner_found=True
                parent=contour
                cv2.drawContours(filled_mask, [inner_contour], 0, 255, thickness=-1)
            child = hierarchy[child][0]




    if inner_found:
        contours, _ = cv2.findContours(
            filled_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        inner_contour = max(contours, key=cv2.contourArea)
        (cx, cy), inner_radius = cv2.minEnclosingCircle(inner_contour)

        cx = int(cx)
        cy = int(cy)
        inner_radius = int(inner_radius)

        best_radius = inner_radius
        best_score = 0

        # 4. Search possible outer radius
        max_radius = inner_radius + 40

        circle_boundary = np.zeros_like(filled_mask)
        cv2.circle(
            circle_boundary,
            (cx, cy),
            inner_radius,
            255,
            thickness=-1
        )
        pre_score=0

        parent_mask = np.zeros_like(red_mask)
        cv2.drawContours(
            parent_mask,
            [parent],
            -1,
            255,
            thickness=-1
        )



        for r in range(inner_radius+1, max_radius):
            cv2.circle(
                circle_boundary,
                (cx, cy),
                r,
                255,
                thickness=2
            )

            overlap = cv2.bitwise_and(
                circle_boundary,
                parent_mask
            )


            score = cv2.countNonZero(overlap)
            diff_score=score-pre_score
            if diff_score!=score and diff_score>best_score:
                best_score = diff_score
                best_radius = r


            pre_score=score




        final_mask = np.zeros_like(filled_mask)

        cv2.circle(
            final_mask,
            (cx, cy),
            best_radius,
            255,
            thickness=-1
        )
        segment = cv2.bitwise_and(before_gau, before_gau, mask=final_mask)
        cv2.imshow("after flood",segment)
    else:
        segment = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)
        cv2.imshow("after flood", segment)


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
    hsv=cv2.merge([h,s,v_clahe])

    lower_blue = np.array([95, 100,40])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_open)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)



    water_blue_mask = watershed_segmentation(image, blue_mask,3,0.55)


    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(water_blue_mask)
    contours, _ = cv2.findContours(water_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        shape = shape_detection(contour, img_area)
        if shape is None:
            continue


        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)
    cv2.imshow("segment",segmented_sign)
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

    mask = yellow_mask.copy()

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
    yellow_mask = yellow_mask | flood_inverse
    cv2.imshow("flood", yellow_mask)

    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(yellow_mask)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sign_found = False
    image_height, image_width = yellow_mask.shape[:2]

    for contour in contours:

        shape = shape_detection(contour, img_area)
        if shape is None:
            continue

        if shape in ("triangle", "rectangle"):
            contour_area_ratio = cv2.contourArea(contour) / img_area
            x, y, width, height = cv2.boundingRect(contour)

            # Ignore masks that are clearly background rather than a sign.
            if contour_area_ratio > 0.75:
                continue

            if shape == "rectangle":
                aspect_ratio = width / float(height)
                if not 0.45 <= aspect_ratio <= 2.20:
                    continue

            sign_found = True

        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    edge_shape = find_yellow_edge_shape(
        closed_edges,
        raw_yellow_mask,
        img_area
    )

    # If normal colour and edge contours fail, suppress background texture and
    # retry. Broken triangle sides are reconstructed from strong lines last.
    if (
        not sign_found
        and (edge_shape is None or len(edge_shape) != 3)
    ):
        smoother_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        fallback_edges = cv2.Canny(smoother_gray, 50, 150)
        fallback_edges = cv2.morphologyEx(
            fallback_edges,
            cv2.MORPH_CLOSE,
            edge_kernel
        )

        smooth_edge_shape = find_yellow_edge_shape(
            fallback_edges,
            raw_yellow_mask,
            img_area
        )

        if smooth_edge_shape is not None and len(smooth_edge_shape) == 3:
            x, y, width, height = cv2.boundingRect(smooth_edge_shape)
            touches_image_edge = (
                x <= 1
                or y <= 1
                or x + width >= image_width - 1
                or y + height >= image_height - 1
            )
            if not touches_image_edge:
                edge_shape = smooth_edge_shape
            else:
                edge_shape = find_yellow_line_triangle(
                    fallback_edges,
                    raw_yellow_mask,
                    img_area
                )
        else:
            edge_shape = find_yellow_line_triangle(
                fallback_edges,
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

        # A two-pixel expansion helps include the thin outer border.
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

def line_intersection(first_line, second_line):
    x1, y1, x2, y2 = first_line
    x3, y3, x4, y4 = second_line

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )
    if abs(denominator) < 1e-6:
        return None

    x = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    y = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    return np.array([x, y], dtype=np.float32)

def find_yellow_line_triangle(edge_mask, raw_yellow_mask, img_area):
    image_height, image_width = edge_mask.shape[:2]
    min_dimension = min(image_height, image_width)

    detected_lines = cv2.HoughLinesP(
        edge_mask,
        1,
        np.pi / 180,
        threshold=max(10, int(min_dimension * 0.12)),
        minLineLength=max(10, int(min_dimension * 0.18)),
        maxLineGap=max(3, int(min_dimension * 0.08))
    )
    if detected_lines is None:
        return None

    line_groups = {
        "left": [],
        "right": [],
        "base": []
    }

    for detected_line in np.asarray(detected_lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(int, detected_line)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        length = float(np.hypot(x2 - x1, y2 - y1))
        line = (x1, y1, x2, y2)

        if abs(angle) <= 20:
            line_groups["base"].append((length, line))
        elif -80 <= angle <= -30:
            line_groups["left"].append((length, line))
        elif 30 <= angle <= 80:
            line_groups["right"].append((length, line))

    for group_name in line_groups:
        line_groups[group_name] = [
            line
            for _, line in sorted(
                line_groups[group_name],
                reverse=True
            )[:30]
        ]

    best_triangle = None
    best_score = -1

    for left_line in line_groups["left"]:
        for right_line in line_groups["right"]:
            top_point = line_intersection(left_line, right_line)
            if top_point is None:
                continue

            for base_line in line_groups["base"]:
                bottom_left = line_intersection(left_line, base_line)
                bottom_right = line_intersection(right_line, base_line)
                if bottom_left is None or bottom_right is None:
                    continue

                if bottom_left[0] > bottom_right[0]:
                    bottom_left, bottom_right = bottom_right, bottom_left

                triangle_points = np.array(
                    [top_point, bottom_left, bottom_right],
                    dtype=np.float32
                )

                if np.any(triangle_points[:, 0] < 0):
                    continue
                if np.any(triangle_points[:, 0] > image_width - 1):
                    continue
                if np.any(triangle_points[:, 1] < 0):
                    continue
                if np.any(triangle_points[:, 1] > image_height - 1):
                    continue

                triangle_height = (
                    (bottom_left[1] + bottom_right[1]) / 2
                    - top_point[1]
                )
                triangle_width = bottom_right[0] - bottom_left[0]

                if triangle_height < image_height * 0.15:
                    continue
                if not bottom_left[0] <= top_point[0] <= bottom_right[0]:
                    continue
                if not 0.50 <= triangle_width / triangle_height <= 2.50:
                    continue

                base_midpoint_x = (
                    bottom_left[0] + bottom_right[0]
                ) / 2
                if (
                    abs(top_point[0] - base_midpoint_x)
                    / triangle_width
                    > 0.15
                ):
                    continue
                if (
                    abs(bottom_left[1] - bottom_right[1])
                    / triangle_height
                    > 0.20
                ):
                    continue

                triangle = np.round(
                    triangle_points
                ).astype(np.int32).reshape(-1, 1, 2)
                area_ratio = cv2.contourArea(triangle) / img_area
                if not 0.07 <= area_ratio <= 0.75:
                    continue

                triangle_mask = np.zeros_like(raw_yellow_mask)
                cv2.drawContours(
                    triangle_mask,
                    [triangle],
                    0,
                    255,
                    thickness=-1
                )
                triangle_area = cv2.countNonZero(triangle_mask)
                if triangle_area == 0:
                    continue

                yellow_inside = cv2.bitwise_and(
                    raw_yellow_mask,
                    triangle_mask
                )
                yellow_ratio = (
                    cv2.countNonZero(yellow_inside)
                    / triangle_area
                )
                if yellow_ratio < 0.03:
                    continue

                triangle_border = np.zeros_like(raw_yellow_mask)
                cv2.drawContours(
                    triangle_border,
                    [triangle],
                    0,
                    255,
                    thickness=2
                )
                border_area = cv2.countNonZero(triangle_border)
                edge_support = (
                    cv2.countNonZero(
                        cv2.bitwise_and(edge_mask, triangle_border)
                    )
                    / border_area
                )
                if edge_support < 0.08:
                    continue

                score = (
                    (2.0 * edge_support)
                    + yellow_ratio
                    + (1.5 * area_ratio)
                )
                if score > best_score:
                    best_triangle = triangle
                    best_score = score

    return best_triangle


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

    epsilon = 0.02 * perimeter
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


def watershed_segmentation(img,mask,iterations,therehold):
    if cv2.countNonZero(mask) == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=iterations)
    cv2.imshow("sure_bg", sure_bg)
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
signs=["RedSigns","BlueSigns","YellowSigns"]
image_files = list((image_path / signs[0]).glob("*.png"))

for item in image_files:
    segment_red_color(item)


