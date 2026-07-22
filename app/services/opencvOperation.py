import cv2
import numpy as np
from pathlib import Path

def segment_red_color(image_path):

    #image receive
    before_gau = cv2.imread(image_path)

    #preprocessing
    if before_gau is None:
        print("Image Not Found")
        return

    image = cv2.GaussianBlur(before_gau, (1,1), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    grid=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    cl_channel = grid.apply(l)
    merged = cv2.merge((cl_channel, a, b))
    output = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


    #segmentation
    b,g,r = cv2.split(output)
    red_condition=(r>b*1.4)&(r>g*1.4)&(r>50)
    red_mask=np.uint8(red_condition)*255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)




    filled_mask = np.zeros_like(red_mask)
    contours, _=cv2.findContours(red_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        shape=shape_detection(contour)
        if shape is None:
            continue
        cv2.drawContours(filled_mask, [contour], 0, 255, thickness=-1)

    segmented_sign = cv2.bitwise_and(before_gau, before_gau, mask=filled_mask)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Segmented Sign", segmented_sign)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
def segment_blue_color(image_path):



    cv2.waitKey(0)
    cv2.destroyAllWindows()
def segment_yellow_color(image_path):



    cv2.waitKey(0)
    cv2.destroyAllWindows()

def shape_detection(contour):
    if len(contour) < 3:
        return None
    perimeter = cv2.arcLength(contour, True)

    area = cv2.contourArea(contour)


    if perimeter == 0:
        return None
    elif area<500:
        return None
    # for polygon
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    # for circle
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    #triangle
    if vertices == 3:
        return 1
    #square
    elif vertices == 4:
        return 2
    #8 vertax
    elif vertices == 8:
        return 3
    #circle
    elif circularity >0.75:
        return 4
    else:
        # shape found not under expected
        return None



BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / ".." / "image" / "ColorInputs" / "RedSigns"
image_files = list(image_path.glob("*.png"))

for i in range(0,6):
    print(image_files[i])
    segment_red_color(image_files[i])

# Run the function with your image path
segment_red_color(image_path)