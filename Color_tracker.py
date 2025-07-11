import cv2
import numpy as np

# input the hsv and color you want to track, itll return the mask
def find_given_color(hsv, color: str):

    match color.lower():
        case "red":
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([160, 100, 100])
            upper2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            return mask1 | mask2

        case "blue":
            lower = np.array([100, 150, 50])
            upper = np.array([130, 255, 255])
            return cv2.inRange(hsv, lower, upper)

        case "green":
            lower = np.array([40, 70, 70])
            upper = np.array([80, 255, 255])
            return cv2.inRange(hsv, lower, upper)

        case "yellow":
            lower = np.array([20, 100, 100])
            upper = np.array([30, 255, 255])
            return cv2.inRange(hsv, lower, upper)

        case "purple":
            lower = np.array([130, 100, 100])
            upper = np.array([155, 255, 255])
            return cv2.inRange(hsv, lower, upper)

        case "orange":
            lower = np.array([10, 150, 100])
            upper = np.array([25, 255, 255])
            return cv2.inRange(hsv, lower, upper)

        case _:
            raise ValueError(f"Unsupported color: {color}")

