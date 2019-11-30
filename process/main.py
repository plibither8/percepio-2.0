import cv2
import numpy as np
import time
import requests
import json
import os
import screeninfo
from PIL import Image
from collections import deque

def recognise(filename):
    payload = {
        'apikey': os.environ['OCR_SPACE_TOKEN'],
        'OCREngine': 2,
        'scale': True
    }
    with open(filename, 'rb') as f:
        r = requests.post(
            'https://api.ocr.space/parse/image',
            files={filename: f},
            data=payload,
        )
    response = json.loads(r.content.decode())
    parsed_results = response["ParsedResults"]

    if len(parsed_results) > 0:
        parsed_text = parsed_results[0]['ParsedText']
        print(parsed_text)
        return parsed_text

    return ''

# Color info
class Color:
    WHITE = [0xFF, 0xFF, 0xFF]
    ORANGE = [0, 97, 0xFF]
    BLUE = [0xFF, 0, 0]
    RED = [0, 0, 255]

# distance between two pixels
def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# main start initiate func
def start(device, flip=0):
    frame_count = 0

    cap = cv2.VideoCapture(device)

    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # gesture matching initialization
    gesture_template = cv2.imread('gesture_template.png')
    gesture_template = cv2.cvtColor(gesture_template, cv2.COLOR_BGR2GRAY)
    (gesture_template, _) = cv2.findContours(gesture_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # skin color segmentation mask
    skin_min = np.array([0, 40, 50], np.uint8)  # HSV mask
    skin_max = np.array([50, 250, 0xFF], np.uint8)  # HSV mask

    # trajectory drawing initialization
    topmost_last    = (int(cap_width / 4), int(cap_height / 4))  # initial position of finger cursor
    traj            = np.array([], np.uint16)
    traj            = np.append(traj, topmost_last)
    dist_pts        = 0
    dist_records    = [dist_pts]

    # finger cursor position low_pass filter
    low_filter_size = 5
    low_filter = deque([topmost_last for i in range(low_filter_size)], low_filter_size)  # filter size is 5

    # gesture_index low_pass filter
    gesture_filter_size = 5
    gesture_matching_filter = deque([0. for i in range(gesture_filter_size)], gesture_filter_size)
    gesture_index_thres = 5

    # some kernels
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2


    screen = screeninfo.get_monitors()[0]
    width, height = int(screen.width / 2), screen.height

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 0, 0)
    cv2.resizeWindow('frame', (width, height))

    text_image = np.zeros((width, height, 3), dtype=np.uint8 )

    cv2.namedWindow('character', cv2.WINDOW_NORMAL)
    cv2.moveWindow('character', width, 0)
    cv2.resizeWindow('character', (width, height))
    cv2.imshow('character', text_image)

    while cap.isOpened():
        if frame_count is not 0:
            frame_count += 1

        # Capture frame-by-frame

        (ret, frame_raw) = cap.read()
        while not ret:
            (ret, frame_raw) = cap.read()
        if flip:
            frame_raw = cv2.flip(frame_raw, 1)
        frame = frame_raw[:round(cap_height), :round(cap_width)]  # ROI of the image

        # Color seperation and noise cancellation at HSV color space
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, skin_min, skin_max)
        res   = cv2.bitwise_and(hsv, hsv, mask=mask)
        res   = cv2.erode(res, kernel, iterations=1)
        res   = cv2.dilate(res, kernel, iterations=1)

        # Canny edge detection at Gray space.
        rgb   = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        gray  = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (11, 11), 0)

        # main function: find finger cursor position & draw trajectory
        (contours, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # find all contours in the image
        if len(contours) > 0:
            biggest_contour = max(contours, key=cv2.contourArea)  # find biggest contour in the image
            if cv2.contourArea(biggest_contour) > 1000:
                topmost = tuple(biggest_contour[biggest_contour[:, :, 1].argmin()][0])  # consider the topmost point of the biggest contour as cursor

                # get matching score, smaller the better by comparing contour of
                # predefined image with that of drawn gesture
                gesture_index = cv2.matchShapes(biggest_contour, gesture_template[0], 1, 0.)

                # obtain average gesture matching index using gesture matching low_pass filter
                gesture_matching_filter.append(gesture_index)
                average_gesture_index = sum(gesture_matching_filter) / gesture_filter_size

                dist_pts = dist(topmost, topmost_last)  # calculate the distance of last cursor position and current cursor position
                if dist_pts < 150:  # filter big position change of cursor
                    cv2.drawContours(rgb, [biggest_contour], 0, (0, 0xFF, 0), 5)
                    low_filter.append(topmost)

                    sum_x = 0
                    sum_y = 0
                    for i in low_filter:
                        sum_x += i[0]
                        sum_y += i[1]

                    topmost = (sum_x // low_filter_size, sum_y // low_filter_size)

                    if gesture_index < gesture_index_thres:
                        traj = np.append(traj, topmost)
                        dist_records.append(dist_pts)

                        if frame_count is 0:
                            frame_count = 1

                    else:
                        traj = np.array([], np.uint16)
                        traj = np.append(traj, topmost_last)

                        dist_pts = 0
                        dist_records = [dist_pts]

                    topmost_last = topmost  # update cursor position

        trace = np.zeros((int(cap_height), int(cap_width), 3), np.uint8)

        for i in range(2, len(dist_records)):
            thickness = int(-0.072 * dist_records[i] + 13)
            cv2.line(
                frame,
                (traj[i * 2 - 2], traj[i * 2 - 1]),
                (traj[i * 2], traj[i * 2 + 1]),
                Color.ORANGE,
                thickness
            )
            cv2.line(
                trace,
                (traj[i * 2 - 2], traj[i * 2 - 1]),
                (traj[i * 2], traj[i * 2 + 1]),
                Color.WHITE,
                thickness
            )

        # pointer circle
        cv2.circle(
            frame,
            topmost_last,
            10,
            Color.BLUE,
            3
        )

        cv2.imshow('frame', frame_raw)

        # every 240/30 = 8seconds
        if frame_count % 240 is 0 and frame_count > 0:
            filename = './output/' + str(time.time()) + '.jpg'
            cv2.imwrite(filename, trace)

            traj = np.array([], np.uint16)
            traj = np.append(traj, topmost_last)

            dist_pts = 0
            dist_records = [dist_pts]

            text_image = np.zeros((width, height, 3), dtype=np.uint8 )
            cv2.putText(
                text_image,
                "Loading...",
                (int(1 * width / 4), int(height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                Color.WHITE,
                3,
                cv2.LINE_AA,
                True
            )
            cv2.imshow('character', text_image)

            text = recognise(filename)
            text_image = np.zeros((width, height, 3), dtype=np.uint8 )

            if (text.strip() == ''):
                cv2.putText(
                    text_image,
                    "No text found",
                    (int(1 * width / 4), int(height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    Color.RED,
                    3,
                    cv2.LINE_AA,
                    True
                )
            else:
                cv2.putText(
                    text_image,
                    text,
                    (int(3 * width / 8), int(height / 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    20,
                    Color.WHITE,
                    10,
                    cv2.LINE_AA,
                    True
                )

            text_image = cv2.flip(text_image, 0)
            cv2.imshow('character', text_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = 2  # if device = 0, use the built-in computer camera
    start(device)
