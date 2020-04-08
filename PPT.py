import cv2 as cv
import numpy as np
import urllib.request

def url_to_image(url):
    with urllib.request.urlopen(url) as url:
        image = np.asarray(bytearray(url.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        return image


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
        hsv = cv.cvtColor(image.copy(), cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        cv.imshow("HSV", mask)
        cv.waitKey(5)


ix, iy = -1, -1
# mouse callback function

pos = []


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result
def mousePosition(event, x, y, flags, param):
    
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(len(pos))
        pos.append((x, y))
        if len(pos) == 4:
            from math import sqrt, pi
            import math
            print(pos)
            a_0 = sqrt((pos[1][0] - pos[2][0]) ** 2 + (pos[1][1] - pos[2][1]) ** 2)
            b_0 = pos[2][0] - pos[1][0]
            angle_0 = math.acos(b_0 / a_0)
            print(angle_0 / pi * 180)
            a_1 = sqrt((pos[1][0] - pos[0][0]) ** 2 + (pos[1][1] - pos[0][1]) ** 2)
            b_1 = pos[1][0] - pos[0][0]
            print(math.acos(b_1 / a_1))
            angle_1 = math.acos(b_1 / a_1) - angle_0
            print(angle_1 / pi * 180)
            x = 250 / a_0 * a_1 * math.cos(angle_1)
            print(x)
            #cv.imshow("rot", rotateImage(image, -angle_0 / pi * 180))

def point(image, hsv):
        from math import pi
        hsv_l = hsv[0]
        hsv_u = hsv[1]
        img = image.copy()
        img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_hsv, hsv_l, hsv_u)
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #cv.imshow("Mask", mask)
        #cv.waitKey(1)
        for cnt in contours:
            # find circle center
            (x, y), radius = cv.minEnclosingCircle(cnt)
            circle_center = (x, y)
            area = cv.contourArea(cnt)
            #cv.circle(img, circle_center, int(radius), (128, 0, 128), 3)
            if area != 0:
                return circle_center
                #print(circle_center)
        #cv.imshow("Img", img)
        #cv.waitKey(0)
hsv = (0, 153, 55), (200, 199, 255)
parameters = aruco.DetectorParameters_create()
url = "https://stepik.org/media/attachments/lesson/284187/test_1_1.jpg"
image = url_to_image(url)
#track(image)

print(point(image, hsv))
while True:
    cv.imshow("image", image)
    cv.setMouseCallback('image', mousePosition)
    cv.waitKey(1)
