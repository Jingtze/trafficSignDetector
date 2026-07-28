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
    cv2.imshow("water", water_red_mask)

    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(water_red_mask)
    contours, _ = cv2.findContours(water_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    cv2.imshow("3. Segmented Sign", segmented_sign)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
        elif shape == "recover":
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.3:
                print(1)
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(filled_mask, ellipse, 255, thickness=-1)

        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)
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
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h,s,v=cv2.split(temp)
    clahe_s=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    clahe_v=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    s_channel=clahe_s.apply(s)

    v_channel=clahe_v.apply(v)

    hsv=cv2.merge((h,s_channel,v))
    hsv2=cv2.merge((h,s_channel,v_channel))
    lower_yellow = np.array([15, 110, 90])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask1 = cv2.inRange(temp, lower_yellow, upper_yellow)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask2 = cv2.inRange(hsv2, lower_yellow, upper_yellow)



    cv2.imshow("yellow mask",yellow_mask)
    cv2.imshow("sv",yellow_mask2)
    cv2.imshow("temp",yellow_mask1)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_open)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_close)


    water_yellow_mask = watershed_segmentation(before_gau, yellow_mask,3,0.50)



    # Segmentation & Shape Checking
    filled_mask = np.zeros_like(water_yellow_mask)
    contours, _ = cv2.findContours(water_yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        shape = shape_detection(contour, img_area)
        if shape is None:
            continue

        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return segmented_sign


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


def watershed_segmentation(img,mask,iterations,therehold):
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

def showInLargeWindow(image1,image2,window_name="Original vs Mask"):
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        # 2. Check heights and resize image2 ONLY if heights actually differ
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    if h1 != h2:
        new_w2 = int(w2 * (h1 / h2))
        image2 = cv2.resize(image2, (new_w2, h1))

    # 3. Stack horizontally (ALWAYS runs regardless of height check)
    stacked_image = np.hstack((image1, image2))

    # 4. Display in resizable large window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)
    cv2.imshow(window_name, stacked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)

    return edges



BASE_DIR = Path(__file__).resolve().parent.parent
image_path = BASE_DIR / "image" / "ColorInputs"
signs=["RedSigns","BlueSigns","YellowSigns"]
image_files = list((image_path / signs[2]).glob("*.png"))

for item in image_files:
    segment_yellow_color(item)


