import cv2
import numpy as np


vc_left = cv2.VideoCapture(0)
# vc_right = cv2.VideoCpature(1)


while 1:
    img = cv2.imread("left_1_Moment.jpg")
    resize_img = cv2.resize(img, (320,240))
    draw_img = resize_img
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 70, 150)
    Hough = cv2.HoughLinesP(canny_img, 2, np.pi/180, 15, np.array([]), )
    for x1,y1,x2,y2 in Hough:
        cv2.line(draw_img, (x1,y1),(x2,y2), color=255,thickness = 3 )

