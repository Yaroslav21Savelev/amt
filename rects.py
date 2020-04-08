#!/usr/bin/env python

import cv2 as cv
import numpy as np
import math


def track(img):
    def nothing(self):
        pass
    window_name = "track_img"
    cv.namedWindow(window_name)

    cv.createTrackbar('UH', window_name, 0, 255, nothing)
    cv.setTrackbarPos('UH', window_name, 255)

    cv.createTrackbar('US', window_name, 0, 255, nothing)
    cv.setTrackbarPos('US', window_name, 255)

    cv.createTrackbar('UV', window_name, 0, 255, nothing)
    cv.setTrackbarPos('UV', window_name, 255)

    # create trackbars for Lower HSV
    cv.createTrackbar('LH', window_name, 0, 255, nothing)
    cv.setTrackbarPos('LH', window_name, 0)

    cv.createTrackbar('LS', window_name, 0, 255, nothing)
    cv.setTrackbarPos('LS', window_name, 0)

    cv.createTrackbar('LV', window_name, 0, 255, nothing)
    cv.setTrackbarPos('LV', window_name, 0)
    while True:
        window_name = "track_img"
        uh = cv.getTrackbarPos('UH', window_name)
        us = cv.getTrackbarPos('US', window_name)
        uv = cv.getTrackbarPos('UV', window_name)
        upper_blue = np.array([uh, us, uv])
        # get current positions of Lower HSCV trackbars
        lh = cv.getTrackbarPos('LH', window_name)
        ls = cv.getTrackbarPos('LS', window_name)
        lv = cv.getTrackbarPos('LV', window_name)
        upper_hsv = np.array([uh, us, uv])
        lower_hsv = np.array([lh, ls, lv])
        print(upper_hsv, lower_hsv)
        hsv = cv.cvtColor(img.copy(), cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        cv.imshow("HSV", mask)
        cv.waitKey(5)

if __name__ == '__main__':
    cv.namedWindow("result")

    hsv_min = np.array((40, 0, 0), np.uint8)
    hsv_max = np.array((135, 62, 109), np.uint8)

    color_blue = (255, 0, 0)
    color_red = (0, 0, 128)

    img = cv.imread("test_1_1.jpg")
    #track(img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)

    cv.imshow("hsv", thresh)
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        center = (int(rect[0][0]), int(rect[0][1]))
        area = int(rect[1][0]*rect[1][1])

        edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

        usedEdge = edge1
        if cv.norm(edge2) > cv.norm(edge1):
            usedEdge = edge2

        reference = (1, 0)  # horizontal edge
        #angle = 180.0/math.pi *  math.acos((reference[0]*usedEdge[0] + reference[1] * usedEdge[1]) / (cv.norm(reference) * cv.norm(usedEdge)))
        angle = 13
        cv.fillPoly(img, pts=[cnt], color=(0, 255, 255))
        #cv.drawContours(img, [box], 0, color_blue, 2)
        #cv.circle(img, center, 5, color_red, 2)
        #cv.putText(img, "%d" % int(angle), (center[0]+20, center[1]-20), cv.FONT_HERSHEY_SIMPLEX, 1, color_red, 2)
    cv.imshow('result', img)
    cv.waitKey(0)

    cv.destroyAllWindows()
