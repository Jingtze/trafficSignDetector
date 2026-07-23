import cv2
import numpy as np
from pathlib import Path



def segment_red_color(image_path):
    # image receive
    before_gau = cv2.imread(str(image_path))

    # preprocessing
    if before_gau is None:
        print("Image Not Found")
        return

    img_area = before_gau.shape[0] * before_gau.shape[1]

    image = cv2.GaussianBlur(before_gau, (7,7), 0)
    

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 60])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)



    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel_close)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel_open)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel_close)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel_open)



    cv2.imshow("red1", mask1)
    cv2.imshow("red2", mask2)

    cv2.waitKey(0)

    red_mask = cv2.bitwise_or(mask1, mask2)
    #red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)
    #red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)

    filled_mask1 = np.zeros_like(mask1)
    filled_mask2 = np.zeros_like(mask2)
    filled_mask3 = np.zeros_like(red_mask)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




    flag1=False
    flag2=False

    for contour in contours1:
        shape = shape_detection(contour,True,img_area)
        if shape is not None:
            flag1 = True
            cv2.drawContours(filled_mask1, [contour], 0, 255, thickness=-1)
    for contour in contours2:
        shape = shape_detection(contour,True,img_area)
        if shape is not None:
            flag2 = True
            cv2.drawContours(filled_mask2, [contour], 0, 255, thickness=-1)
    if not flag1 and not flag2:

        contours3, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours3:
            shape = shape_detection(contour,False,img_area)
            if shape is not None:
                cv2.drawContours(filled_mask3, [contour], 0, 255, thickness=-1)


    if flag1:
        segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask1)
        print(1)
        cv2.imshow("Segmented Sign",segmented_sign)
    elif flag2:
        segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask2)
        print(2)
        cv2.imshow("Segmented Sign",segmented_sign)
    else:
        segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask3)
        print(3)
        cv2.imshow("Segmented Sign",segmented_sign)
        cv2.waitKey(0)


    cv2.imshow("Red Mask", red_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_blue_color(image_path):
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_yellow_color(image_path):
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def shape_detection(contour,strict, img_area=10000 ):
    if len(contour) < 3:
        return None

    perimeter = cv2.arcLength(contour, True)

    area = cv2.contourArea(contour)



    if perimeter == 0:
        return None

    min_area = max(500, img_area * 0.001)
    if area < min_area:
        return None

    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    if not cv2.isContourConvex(approx) and strict:
         return None
     #for circle
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # Relaxed shape constraints
    if vertices == 3:
        return 1  # triangle
    elif vertices == 4:
        return 2  # square/rectangle
    elif vertices == 8:
        return 3  # octagon
    elif circularity > 0.80:
        return 4  # circle/ellipse
    else:
        # shape found not under expected
        return None


BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / ".." / "image" / "ColorInputs" / "RedSigns"
image_files = list(image_path.glob("*.png"))

for i in range(0, max(6, len(image_files))):
    print(image_files[i])
    segment_red_color(image_files[i])