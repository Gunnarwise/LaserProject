import cv2
import numpy as np

## VARIABLES ##
color_to_track = "blue" # set default tracking color
modes = ["default", "targeting", "smooth targeting"]
mode_index = 0
mode = modes[mode_index]


smoothed_cx, smoothed_cy = None, None
smoothing_factor = 0.2


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






# start capturing webcam (0 = default webcam)
cap = cv2.VideoCapture(1)

while True:
    # read = bool if read worked or not
    # frame = array storing pixels from the frame in BGR (blue green red)
    read, frame = cap.read()

    # if read failed, end program
    if not read:
        break

    # Flip image horizontally for natural movement (optional)
    frame = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)




    # use function to detect given color
    mask = find_given_color(hsv, color_to_track)
    # erode and dilate to clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if mode == "default":
        for contour in contours:
            # get area of contour
            area = cv2.contourArea(contour)

            # show squares with area bigger than 500
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2

                # Draw rectangle and center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"Target: ({cx}, {cy})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



    elif mode == "smooth targeting":

        # Filter out small/noisy contours
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]

        if large_contours:

            # Get the largest valid contour
            largest = max(large_contours, key=cv2.contourArea)

            M = cv2.moments(largest)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Smooth motion
                if smoothed_cx is None:
                    smoothed_cx, smoothed_cy = cx, cy
                else:
                    smoothed_cx = int((1 - smoothing_factor) * smoothed_cx + smoothing_factor * cx)
                    smoothed_cy = int((1 - smoothing_factor) * smoothed_cy + smoothing_factor * cy)


                # Draw crosshair
                color = (0, 0, 255)
                l = 20
                cv2.line(frame, (smoothed_cx - l, smoothed_cy), (smoothed_cx + l, smoothed_cy), color, 4)
                cv2.line(frame, (smoothed_cx, smoothed_cy - l), (smoothed_cx, smoothed_cy + l), color, 4)
                cv2.circle(frame, (smoothed_cx, smoothed_cy), 5, (255, 255, 255), -1)
                cv2.putText(frame, f"({smoothed_cx},{smoothed_cy})", (smoothed_cx + 10, smoothed_cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    elif mode == "targeting":

        # Filter out small/noisy contours
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]

        if large_contours:

            # Get the largest valid contour
            largest = max(large_contours, key=cv2.contourArea)

            M = cv2.moments(largest)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])


                # Draw crosshair
                color = (0, 0, 255)
                l = 20
                cv2.line(frame, (cx - l, cy), (cx + l, cy), color, 4)
                cv2.line(frame, (cx, cy - l), (cx, cy + l), color, 4)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


    # show color and mode
    cv2.putText(frame, f"Tracking: {color_to_track.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    cv2.putText(frame, f"Mode: {mode}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    # Window that shows everything + boxes around color being tracked
    cv2.imshow("Balloon Tracker", frame)

    # window that is black and white (white = color thats being tracked)
    #cv2.imshow("Mask", mask)

    # key binds for exiting and changing color
    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):
        break
    elif key == ord('r'):
        color_to_track = "red"
    elif key == ord('b'):
        color_to_track = "blue"
    elif key == ord('y'):
        color_to_track = "yellow"
    elif key == ord('g'):
        color_to_track = "green"
    elif key == ord('o'):
        color_to_track = "orange"
    elif key == ord('p'):
        color_to_track = "purple"
    elif key == ord('m'):
        mode_index = (mode_index + 1) % len(modes)
        mode = modes[mode_index]



# Cleanup
cap.release()
cv2.destroyAllWindows()