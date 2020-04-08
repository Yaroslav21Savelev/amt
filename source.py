import cv2 as cv
cv2 = cv
import numpy as np
import urllib.request


def url_to_image(url):
    with urllib.request.urlopen(url) as url:
        image = np.asarray(bytearray(url.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        return image

'''
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
window_name = "track_img"

    


url = ["https://stepik.org/media/attachments/lesson/284187/test_1_1.jpg",
"https://stepik.org/media/attachments/lesson/284187/example_2__-50_-70_.jpg",
"https://stepik.org/media/attachments/lesson/284187/example_3__190_50_.jpg",
"https://stepik.org/media/attachments/lesson/284187/example_4__250_125_.jpg",
"https://stepik.org/media/attachments/lesson/284187/example_5__50_175_.jpg"]
images = []
for i in url:
    print(i)
    images.append(url_to_image(i))
while True:
    for i in range(len(url)):
        img = images[i]
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
        cv.imshow(str(i), mask)
        cv.waitKey(5)
        
'''
def dot(image):
    RED = (0, 128, 212), (255, 237, 255)
    img = image.copy()
    img = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, RED[0], RED[1])
    #cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for c in contours:
        # compute the center of the contour
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv.putText(img, "center", (cX - 20, cY - 20),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #cv.imshow("Circle", img)
        return cX, cY

def inside(image):
    from math import pi
    img = image.copy()
    #BLACK = (0, 0, 0), (255, 179, 148)
    BLACK = (0, 0, 0), (255, 255, 148)
    #img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_ = cv.inRange(img_hsv, BLACK[0], BLACK[1])

    contours, hierarchy = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hie = -1
    old = []
    j = 0
    out = np.zeros((3, 4), float)
    for i in contours:
        hie += 1
        size = cv2.contourArea(i)
        rect = cv2.minAreaRect(i)
        
        if 100 < size < 10000:
            if len(i) < 5:
                continue
            bEllipse = cv.fitEllipse(i)
            hull = cv.convexHull(i, True)
            hull = cv.approxPolyDP(hull, 15, True)
            if (not cv.isContourConvex(hull)):
                continue
            #angle - bEllipse[2]
            
            if len(hull) == 4:
                if hierarchy[0][hie][2] != -1:
                    n = hierarchy[0][hie][2]
                    while hierarchy[0][n][2] != -1:
                        n = hierarchy[0][n][2]
                    cnt = contours[n]
                    size = cv2.contourArea(cnt)
                    rect = cv2.minAreaRect(cnt)
                    if size < 100.0 or n in old:
                        continue
                    #print(size)
                    j += 1
                    old.append(n)

                    (x, y), radius = cv.minEnclosingCircle(contours[n])
                    circle_center = (x, y)
                    area = cv.contourArea(contours[n])
                    #cv.circle(img, circle_center, int(radius), (128, 0, 128), 3)
                    cA = pi * (radius ** 2) / area

                    #cX, cY = int(bEllipse[0][0]), int(bEllipse[0][1])
                    cX, cY = x, y
                    #cv.drawContours(img, [contours[n]], -1, (0, 255, 255), 2)
                    #cv.circle(img, (cX, cY), 7, (255, 255, 255), -1)
                    #cv.putText(img, "center" + str(j), (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #cv2.imshow('Mask', mask_)
                    #cv2.imshow('IMG', img)  
                    #cv2.waitKey(1)
                    out[j - 1] = [cX, cY, cA, j]
    out = out[out[:, 2].argsort()]

    out = np.delete(out, 2, 1).astype(int)
    if(out[0][0] - out[1][0])**2 + (out[0][1] - out[1][1])**2 > (out[0][0] - out[2][0])**2 + (out[0][1] - out[2][1])**2:
        out[[1, 2]] = out[[2, 1]]
    return out


urls = ["https://stepik.org/media/attachments/lesson/284187/test_1_1.jpg", 
        "https://stepik.org/media/attachments/lesson/284187/example_1__125_125_.jpg",
       "https://stepik.org/media/attachments/lesson/284187/example_2__-50_-70_.jpg",
       "https://stepik.org/media/attachments/lesson/284187/example_3__190_50_.jpg",
       "https://stepik.org/media/attachments/lesson/284187/example_4__250_125_.jpg",
       "https://stepik.org/media/attachments/lesson/284187/example_5__50_175_.jpg"]
images = [
    "test_1_1.jpg",
    "example_1__125_125_.jpg",
    "example_2__-50_-70_.jpg",
    "example_3__190_50_.jpg",
    "example_4__250_125_.jpg",
    "example_5__50_175_.jpg"
]


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(
      image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result
url = input()
img = url_to_image(url)
#img = cv.imread(url)
red = dot(img)
marker = inside(img)
cv.circle(img, (red[0], red[1]), 7, (255, 0, 0), -1)
cv.putText(img, str(red[0]) + " "+ str(red[1]), (red[0] - 20, red[1] - 20),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv.circle(img, (marker[0][0], marker[0][1]), 7, (0, 0, 255), -1)
cv.putText(img, str(marker[0][0]) + " " + str(marker[0][1]), (marker[0][0] - 20, marker[0][1] - 20),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv.circle(img, (marker[1][0], marker[1][1]), 7, (255, 255, 255), -1)
cv.putText(img, str(marker[1][0]) + " " + str(marker[1][1]), (marker[1][0] - 20, marker[1][1] - 20),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv.circle(img, (marker[2][0], marker[2][1]), 7, (0, 255, 255), -1)
cv.putText(img, str(marker[2][0]) + " " + str(marker[2][1]), (marker[2][0] - 20, marker[2][1] - 20),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
points = []
for i in marker:
    points.append((i[0], i[1]))
xp = red[0]
yp = red[1]
xc = (points[0][0] + points[2][0]) / 2
yc = (points[0][1] + points[2][1]) / 2
points.append((int(2*xc-points[1][0]), int(2*yc-points[1][1])))

A = np.array([[points[3][0], points[3][1], 1,            0,            0, 0,            0,            0, 0,    0,    0,  0],
            [points[2][0], points[2][1], 1,            0,            0,
            0,            0,            0, 0, -250,    0,  0],
            [points[1][0], points[1][1], 1,            0,            0,
            0,            0,            0, 0,    0, -250,  0],
            [points[0][0], points[0][1], 1,            0,            0,
            0,            0,            0, 0,    0,    0,  0],
            [0,            0, 0, points[3][0], points[3][1],
                1,            0,            0, 0,    0,    0,  0],
            [0,            0, 0, points[2][0], points[2][1],
                1,            0,            0, 0, -250,    0,  0],
            [0,            0, 0, points[1][0], points[1][1],
                1,            0,            0, 0,    0,    0,  0],
            [0,            0, 0, points[0][0], points[0][1],
                1,            0,            0, 0,    0,    0,  0],
            [0,            0, 0,            0,            0, 0,
                points[3][0], points[3][1], 1,    0,    0,  0],
            [0,            0, 0,            0,            0, 0,
                points[2][0], points[2][1], 1,   -1,    0,  0],
            [0,            0, 0,            0,            0, 0,
                points[1][0], points[1][1], 1,    0,   -1,  0],
            [0,            0, 0,            0,            0, 0,
                points[0][0], points[0][1], 1,    0,    0, -1]
            ])

b = np.array([0, 0, 0, 0, 250, 0, 0, 0, 1, 0, 0, 0])

c = np.linalg.solve(A, b)

x = int((c[0]*xp + c[1]*yp + c[2]) / (c[6]*xp + c[7]*yp + c[8]))
y = int((c[3]*xp + c[4]*yp + c[5]) / (c[6]*xp + c[7]*yp + c[8]))

print(x, y)
cv.imshow("Image", img)
cv.waitKey(0)

