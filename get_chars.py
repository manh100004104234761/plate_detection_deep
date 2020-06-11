import possibleChars
import cv2
import math
import numpy as np
import os
import argparse

MIN_PIXEL_WIDTH = 1
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.1
MAX_ASPECT_RATIO = 0.9

MIN_PIXEL_AREA = 63

MAX_CHANGE_IN_AREA = 0.5

MIN_CONTOUR_AREA = 15


def preprocess(img_original):
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # img_max_contract_gray = maxmize_contrast(img_gray)

    height, width = img_gray.shape

    # img_blurred = np.zeros((height, width, 1), np.uint8)

    img_blurred = cv2.GaussianBlur(
        img_gray, (3, 3), 0)

    img_thresh = cv2.adaptiveThreshold(img_blurred, 255.0,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       3,
                                       1)

    return img_gray, img_thresh


def find_chars(img_thresh):
    list_of_chars = []
    img_thresh_cp = img_thresh.copy()
    cv2.rectangle(img_thresh_cp, (0, 0), (
        img_thresh.shape[1],
        img_thresh.shape[0]),
        (0, 0, 0), 3)
    contours, _ = cv2.findContours(
        img_thresh_cp, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possibleChar = possibleChars.PossibleChar(contour)
        if check_if_char(possibleChar):
            list_of_chars.append(possibleChar)
    list_of_chars.sort(
        key=lambda matching_char: matching_char.int_center_x)
    cropped = []
    for i in range(0, len(list_of_chars)):
        [int_x, int_y, int_w, int_h] = [list_of_chars[i].int_rect_x,
                                        list_of_chars[i].int_rect_y,
                                        list_of_chars[i].int_rect_w,
                                        list_of_chars[i].int_rect_h]
        img_roi = img_thresh[int_y:int_y+int_h, int_x:int_x+int_w]
        img_roi_resized = cv2.resize(
            img_roi, (28, 28), interpolation=cv2.INTER_AREA)
        cropped.append(img_roi_resized)
    return cropped


def check_if_char(possibleChar):
    if (possibleChar.int_rect_area > MIN_PIXEL_AREA
        and possibleChar.int_rect_w > MIN_PIXEL_WIDTH
        and possibleChar.int_rect_h > MIN_PIXEL_HEIGHT
        and MIN_ASPECT_RATIO < possibleChar.flt_aspect_ratio
            and possibleChar.flt_aspect_ratio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
