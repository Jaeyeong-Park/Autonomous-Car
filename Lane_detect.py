import math
import cv2
import numpy as np
import random
import time
import socket

def ToGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def setRoi(img, roi):
    mask = np.zeros_like(img)
    if len(img.shape) <= 2:
        color = 255
    else:
        color = (255,255,255)
    cv2.fillPoly(mask, roi, color)
    roi_img = cv2.bitwise_and(img, mask)
    return roi_img

def canny(img, low_threshold = 70, high_threshold = 150):
    return cv2.Canny(img, low_threshold, high_threshold)

def Hough(img, rho=2, thetha = np.pi/180, threshold = 15, minLineLength = 3):
    return cv2.HoughLinesP(img, rho, thetha, threshold, np.array([]), minLineLength)

def Hough_To_List(Lines):
    detected_line = []
    for n in Lines:
        for x1,y1,x2,y2 in n:
            a = [int(x1), int(y1)]
            b = [int(x2), int(y2)]
            detected_line.append(a)
            detected_line.append(b)

    return detected_line

def compute_distance(par, random_pixel):
    return abs(par[0] * random_pixel[0] + par[1] * random_pixel[1] + par[2]) / math.sqrt(par[0] ** 2 + par[1] ** 2)

def compute_slope(Line_array):
    # y = mx + n
    m = (Line_array[1][1] - Line_array[0][1]) / (Line_array[1][0] - Line_array[0][0])
    n = Line_array[0][1] - m * Line_array[0][0]
    # ax + by + c = 0
    a, b, c = m, -1, n
    par = np.array([a,b,c])
    return par

def Random_sample(Line_array):
    if Line_array.__sizeof__() <= 72 :
        return 1
    sample1 = random.choice(Line_array)
    sample2 = random.choice(Line_array)

    m = 0
    Loop_cnt = 0

    if sample1[0] == sample2[0] or sample1[1] == sample2[1] or m == 0:
        sample2 = random.choice(Line_array)
        sample1 = random.choice(Line_array)
        if sample1[0] != sample2[0]:
            m = (sample2[1] - sample1[1]) / (sample2[0] - sample1[0])
        if Loop_cnt > 20 and Line_array.__sizeof__() <= 400:
            return 1
        Loop_cnt += 1
        while sample1[0] == sample2[0] or sample1[1] == sample2[1] or m < -0.25 or m > 0.25:
            sample2 = random.choice(Line_array)
            sample1 = random.choice(Line_array)
            if sample1[0] != sample2[0]:
                m = (sample2[1] - sample1[1]) / (sample2[0] - sample1[0])
            if Loop_cnt > 20 and Line_array.__sizeof__() <= 400:
                return 1
            Loop_cnt += 1

    Random_Line = []
    Random_Line.append(sample1)
    Random_Line.append(sample2)
    return Random_Line

def Rand_pixel(Line_array):
    return random.choice(Line_array)

def Ransac(Line_array):
    # resize 필요하면 넣기
    global Result_par
    global Result_Line
    max = 0

    for n in range(1, 17):
        sample = Random_sample(Line_array)
        if sample != 1:
            par = compute_slope(sample)
            cnt = 0
            for x in range(1, int(len(Line_array))):
                pixel = Rand_pixel(Line_array)
                dist = compute_distance(par, pixel)
                if dist < 2:
                    cnt += 1

            if max < cnt:
                max = cnt
                Result_par = par
                Result_Line = sample
        else:
            Result_par = 1

    return Result_par

def draw_line(img, par):
    # y좌표 수정하기
    new_x1 = int((0 - par[2]) / par[0])
    new_x2 = int((320 - par[2]) / par[0])
    cv2.line(img, (new_x1, 0), (new_x2, 320), color=[0,0,255], thickness = 6)

def draw_line_mask(img, par):
    # y좌표 수젇하기
    new_x1 = int((0 - par[2]) / par[0])
    new_x2 = int((320 - par[2]) / par[0])
    cv2.line(img, (new_x1, 0), (new_x2, 320), color=(255,255,255), thickness=37)

def main():
    #vc_left = cv2.VideoCapture("left_2.avi")
    #vc_right = cv2.VideoCapture("right_2.avi")


    vc_left = cv2.VideoCapture(0)
    vc_right = cv2.VideoCapture(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 8081))

    global next_Line_mask_Left
    global next_Line_mask_Right

    global before_Rpar
    global before_Lpar

    global Left_error_cnt
    global Right_error_cnt

    Left_cnt = 0
    Right_cnt = 0
    frame_cnt = 1
    back_to_first = 0

    Left_error_cnt = 0
    Right_error_cnt = 0
    global steering_value
    steering_value=0

   ##################################################################################
    # img = cv2.imread("left_1_Moment.jpg")
    # img2 =cv2.imread("left_1_Moment.jpg")
    # img3 = cv2.imread("left_1_Moment.jpg")
    # img4 = cv2.imread("left_1_Moment.jpg")
    # resize_img = cv2.resize(img, (320, 240))
    # resize_img2 = cv2.resize(img2, (320, 240))
    # resize_img3 = cv2.resize(img3, (320, 240))
    # resize_img4 = cv2.resize(img4, (320, 240))
    # draw_img = resize_img2
    # draw_img2 = resize_img3
    # gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    # canny_img = cv2.Canny(gray_img, 70, 150)
    #
    # mask = np.zeros_like(resize_img4)
    #
    # roi = np.array([[(0,65), (0, 190), (320, 190), (320, 65)]] ,dtype = np.int32)
    # roi_img = setRoi(canny_img, roi)
    #
    # Hough = cv2.HoughLinesP(roi_img, 2, np.pi / 180, 15, np.array([]) )
    # for line in Hough:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(draw_img, (x1, y1), (x2, y2), color=255, thickness=3)
    # Lines = Hough_To_List(Hough)
    # par = Ransac(Lines)
    # draw_line(draw_img2, par)
    # draw_line_mask(mask, par)
    # roi2 = cv2.bitwise_and(resize_img4, mask)
    #
    # cv2.imshow("resize",resize_img)
    # cv2.imshow("gray",gray_img)
    # cv2.imshow("canny",canny_img)
    # cv2.imshow("roi",roi_img)
    # cv2.imshow("draw_img",draw_img)
    # cv2.imshow("ransac",draw_img2)
    # cv2.imshow("draw_line_mask",mask)
    # cv2.imshow("roi2", roi2)
    #
    # cv2.imwrite("resize.jpg",resize_img)
    # cv2.imwrite("gray.jpg",gray_img)
    # cv2.imwrite("canny.jpg",canny_img)
    # cv2.imwrite("roi.jpg",roi_img)
    # cv2.imwrite("draw_img.jpg",draw_img)
    # cv2.imwrite("ransac.jpg",draw_img2)
    # cv2.imwrite("draw_line_mask.jpg",mask)
    # cv2.imwrite("roi2.jpg", roi2)
    # cv2.waitKey(0)
   ###############################################################################




    while 1:
        if back_to_first == 0:
            # 왼쪽 이미지
            ret, img_left = vc_left.read()
            # 오른쪽 이미지
            ret2, img_right = vc_right.read()

            # 이미지 전처리
            re_img_left = cv2.resize(img_left, (320, 240))
            re_img_right = cv2.resize(img_right, (320, 240))

            blur_img_left = cv2.GaussianBlur(re_img_left, (5, 5), 0)
            blur_img_right = cv2.GaussianBlur(re_img_right, (5, 5), 0)

            gray_img_left = ToGray(blur_img_left)
            gray_img_right = ToGray(blur_img_right)

            canny_img_left = canny(gray_img_left)
            canny_img_right = canny(gray_img_right)

            next_Line_mask_Left = np.zeros_like(canny_img_left)
            next_Line_mask_Right = np.zeros_like(canny_img_right)

            height, width = re_img_left.shape[:2]

            # ROI영역 지정. 영상별로 좌표 다르게 해야함!
            First_Roi_left = np.array([[(0,65), (0, 190), (width, 190), (width, 65)]]
                                 ,dtype = np.int32)
            First_Roi_right = np.array([[(0,65), (0, 180), (width, 180), (width, 65)]]
                                 ,dtype = np.int32)

            # 엣지 이미지에 ROI 설정.
            roi_img_left = setRoi(canny_img_left, First_Roi_left)
            roi_img_right = setRoi(canny_img_right, First_Roi_right)

            Left_Lines = Hough(roi_img_left)
            Right_Lines = Hough(roi_img_right)

            if Left_Lines.__sizeof__() > 20:
                Left_Line = Hough_To_List(Left_Lines)
                Lpar = Ransac(Left_Line)
                if Lpar.__sizeof__() == 120:
                    draw_line(re_img_left, Lpar)
                    draw_line_mask(next_Line_mask_Left, Lpar)
                    #print("Lpar:", Lpar[0])
                    print("Lpar[2]:", Lpar[2])
                    if abs(Lpar[0]) > 0.08:
                        Left_cnt += 1
                        if Left_cnt > 5 :#or (Lpar[2] <160 and Lpar[2] > 210):
                            if Lpar[0] > 0:
                                steering_value=1
                                print("Turn Right")
                            else:
                                steering_value=2
                                print("Turn Left")
                        else:
                            steering_value=0
                            print("Go Straight")
                            if Lpar[2] <= 110:
                                steering_value = 2
                                print("NO! TURN LEFT!")
                            elif Lpar[2] >= 150:
                                steering_value = 1
                                print("NO! TURN Right!")

                        print(Left_cnt)
                    else:
                        Left_cnt = 0
                        steering_value=0
                        print("Go Straight")
                        print(Left_cnt)
                        if Lpar[2] <= 110:
                            steering_value = 2
                            print("NO! TURN LEFT!")
                        elif Lpar[2] >= 150:
                            steering_value = 1
                            print("NO! TURN Right!")
                    # before_Lpar = Lpar
            # else:
            #     draw_line(re_img_left, before_Lpar)
            #     draw_line_mask(next_Line_mask_Left, before_Lpar)
            #     Left_error_cnt += 1
            #     # Lpar = before_Lpar

            else:
                if Right_Lines.__sizeof__() > 20:
                    print("Change to Rpar")
                    Right_Line = Hough_To_List(Right_Lines)
                    Rpar = Ransac(Right_Line)
                    if Rpar.__sizeof__() == 120:
                        draw_line(re_img_right, Rpar)
                        draw_line_mask(next_Line_mask_Right, Rpar)
                        print("Rpar:", Rpar[0])
                        if abs(Rpar[0]) > 0.08:
                            Right_cnt += 1
                            if Right_cnt > 5:
                                if Rpar[0] > 0:
                                    steering_value=1
                                    print("Turn Right")
                                else:
                                    steering_value=2
                                    print("Turn Left")
                            else:
                                steering_value=0
                                print("Go Straight")
                                if Rpar[2] <= 110:
                                    steering_value = 1
                                    print("NO! TURN Right!")
                                elif Rpar[2] >= 150:
                                    steering_value = 2
                                    print("NO! TURN Left!")
                            print(Right_cnt)

                        else:
                            Right_cnt = 0
                            steering_value=0
                            print(Right_cnt)
                            print("Go Straight")
                            if Rpar[2] <= 110:
                                steering_value = 1
                                print("NO! TURN Right!")
                            elif Rpar[2] >= 150:
                                steering_value = 2
                                print("NO! TURN Left!")
                else:
                    steering_value=0
                    print("no Lpar, Rpar")
                    print("Go Straight")

                    # before_Rpar = Rpar
            # else:
            #     draw_line(re_img_right, before_Rpar)
            #     draw_line_mask(next_Line_mask_Right, before_Rpar)
            #     Right_error_cnt += 1
            #     # Rpar = before_Rpar



            frame_cnt += 1

            # 15프레임마다 Roi에서 차선 구하기 위한 카운트
            back_to_first += 1

            cv2.imshow('left', re_img_left)
            cv2.imshow('right', re_img_right)
            sock.sendto(str(steering_value).encode(), ("127.0.0.1", 8080))
            if cv2.waitKey(33) == 27:
                return 0
        else:
            # 왼쪽 이미지
            ret, img_left = vc_left.read()
            # 오른쪽 이미지
            ret, img_right = vc_right.read()

            # 이미지 전처리
            re_img_left = cv2.resize(img_left, (320, 240))
            re_img_right = cv2.resize(img_right, (320, 240))

            blur_img_left = cv2.GaussianBlur(re_img_left, (5, 5), 0)
            blur_img_right = cv2.GaussianBlur(re_img_right, (5, 5), 0)

            gray_img_left = ToGray(blur_img_left)
            gray_img_right = ToGray(blur_img_right)

            canny_img_left = canny(gray_img_left)
            canny_img_right = canny(gray_img_right)

            height, width = re_img_left.shape[:2]

            # 엣지 이미지에 ROI 설정.
            roi_img_left = cv2.bitwise_and(canny_img_left, next_Line_mask_Left)
            roi_img_right = cv2.bitwise_and(canny_img_right, next_Line_mask_Right)

            # 다음 프레임을 위한 검정색 마스크 이미지 설정
            next_Line_mask_Left = np.zeros_like(canny_img_left)
            next_Line_mask_Right = np.zeros_like(canny_img_right)

            Left_Lines = Hough(roi_img_left)
            Right_Lines = Hough(roi_img_right)

            if Left_Lines.__sizeof__() > 20:
                Left_Line = Hough_To_List(Left_Lines)
                Lpar = Ransac(Left_Line)
                if Lpar.__sizeof__() == 120:
                    draw_line(re_img_left, Lpar)
                    draw_line_mask(next_Line_mask_Left, Lpar)
                    #print("Lpar:", Lpar[0])
                    print("Lpar[2]:", Lpar[2])
                    if abs(Lpar[0]) > 0.08:
                        Left_cnt += 1
                        if Left_cnt > 5:# or  (Lpar[2] >160 and Lpar[2] < 210):
                            if Lpar[0] > 0:
                                steering_value=1
                                print("Turn Right")
                            else:
                                steering_value=2
                                print("Turn Left")
                        else:
                            steering_value=0
                            print("Go Straight")
                            if Lpar[2] <= 110:
                                steering_value = 2
                                print("NO! TURN LEFT!")
                            elif Lpar[2] >= 150:
                                steering_value = 1
                                print("NO! TURN Right!")
                        print(Left_cnt)
                    else:
                        Left_cnt = 0
                        steering_value=0
                        print("Go Straight")
                        if Lpar[2] <= 110:
                            steering_value = 2
                            print("NO! TURN LEFT!")
                        elif Lpar[2] >= 150:
                            steering_value = 1
                            print("NO! TURN Right!")
                        print(Left_cnt)
                    # before_Lpar = Lpar

            # else:
            #     draw_line(re_img_left, before_Lpar)
            #     draw_line_mask(next_Line_mask_Left, before_Lpar)
            #     Left_error_cnt += 1
            #     # Lpar = before_Lpar
            else:
                if Right_Lines.__sizeof__() > 20:
                    print("Change to Rpar")
                    Right_Line = Hough_To_List(Right_Lines)
                    Rpar = Ransac(Right_Line)
                    if Rpar.__sizeof__() == 120:
                        draw_line(re_img_right, Rpar)
                        draw_line_mask(next_Line_mask_Right, Rpar)
                        print("Rpar:", Rpar[0])
                        if abs(Rpar[0]) > 0.08:
                            Right_cnt += 1
                            if Right_cnt > 5:
                                if Rpar[0] > 0:
                                    steering_value=1
                                    print("Turn Right")
                                else :
                                    steering_value=2
                                    print("Turn Left")
                            else:
                                steering_value=0
                                print("Go Straight")
                                if Rpar[2] <= 110:
                                    steering_value = 1
                                    print("NO! TURN Right!")
                                elif Rpar[2] >= 150:
                                    steering_value = 2
                                    print("NO! TURN Left!")
                            print(Right_cnt)
                        else:
                            Right_cnt = 0
                            steering_value=0
                            print(Right_cnt)
                            print("Go Straight")
                            if Rpar[2] <= 110:
                                steering_value = 1
                                print("NO! TURN Right!")
                            elif Rpar[2] >= 150:
                                steering_value = 2
                                print("NO! TURN Left!")
                else:
                    steering_value=0
                    print("no Lpar, Rpar")
                    print("Go Straight")

                    # before_Rpar = Rpar
            # else:
            #     draw_line(re_img_right, before_Rpar)
            #     draw_line_mask(next_Line_mask_Right, before_Rpar)
            #     Right_error_cnt += 1
            #     # Rpar = before_Rpar



            frame_cnt += 1

            # 15프레임마다 Roi에서 차선 구하기 위한 카운트
            back_to_first += 1

            if back_to_first == 15 or Left_error_cnt > 2 or Right_error_cnt > 2:
                back_to_first = 0
                if Left_error_cnt > 2 or Right_error_cnt > 2:
                    print('못 찾아서 처음부터')

            cv2.imshow('left', re_img_left)
            cv2.imshow('right', re_img_right)

            sock.sendto(str(steering_value).encode(), ("127.0.0.1",8080))

            print("steering : ", steering_value)


            if cv2.waitKey(33) == 27:
                return 0



if __name__ == '__main__':
    main()