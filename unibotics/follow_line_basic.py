"""Follow Line, simple P controller (Session 1 solution).

Paste this file into the Unibotics Follow Line editor and press Run.
It is also shown inside notebooks/s1_follow_line.py (step 3.1).
"""
import HAL
import WebGUI
import Frequency
import cv2
import numpy as np

# ---------------- Parameters: tune these! ----------------
V = 2.0                 # linear speed
KP = 1.0                # proportional gain
ROI = (0.75, 0.85)      # band of the image used (fractions of height)
LOWER_RED = np.array([0, 120, 70])
UPPER_RED = np.array([10, 255, 255])
# ----------------------------------------------------------

last_error = 0.0

while True:
    Frequency.tick(50)

    # SENSE
    img = HAL.getImage()
    h, w = img.shape[:2]

    # PERCEIVE
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    top, bottom = int(ROI[0] * h), int(ROI[1] * h)
    m = cv2.moments(mask[top:bottom, :])

    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        error = (cx - w / 2) / (w / 2)  # in [-1, 1]
        last_error = error
        cv2.circle(img, (int(cx), (top + bottom) // 2), 6, (0, 255, 0), -1)
    else:
        error = last_error  # line lost: keep turning the same way

    # CONTROL
    HAL.setV(V)
    HAL.setW(-KP * error)

    # DEBUG
    cv2.rectangle(img, (0, top), (w - 1, bottom), (255, 200, 0), 2)
    cv2.line(img, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    WebGUI.showImage(img)
