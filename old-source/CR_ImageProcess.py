import cv2
import numpy as np
import math
from calcleaf.mause import mouse_pointlog as mp
from calcleaf.gui import area_select
from trackers import Trucker


Green_lowH = 25
Green_highH = 80
Green_lowS = 80


def square_exchange(img, vertax_pts, sq_size=600):
    pts1 = np.float32(vertax_pts)
    pts2 = np.float32([[sq_size, sq_size], [sq_size, 0], [0, 0], [0, sq_size]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    extracted_img = cv2.warpPerspective(img, M, (sq_size, sq_size))
    return extracted_img


def CircleFitting(x, y):
    """最小二乗法による円フィッティングをする関数
        input: x,y 円フィッティングする点群

        output  cxe 中心x座標
                cye 中心y座標
                re  半径

        参考
        一般式による最小二乗法（円の最小二乗法）　画像処理ソリューション
        http://imagingsolution.blog107.fc2.com/blog-entry-16.html
    """

    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    T = np.linalg.inv(F).dot(G)

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)
    re = math.sqrt(cxe ** 2 + cye ** 2 - T[2])
    # print (cxe,cye,re)
    return (round(cxe), round(cye), round(re))


def calc_leaf_CR(File_name,resized_ratio = 0.3):
    img = cv2.imread(File_name, cv2.IMREAD_UNCHANGED)
    width = img.shape[1]
    height = img.shape[0]

    img2 = cv2.resize(img, (int(width * resized_ratio), int(height * resized_ratio)))
    window_title = "Area_Select"
    cv2.namedWindow(window_title)

    limit = 4
    pt_1 = mp.PointLog(limit)
    cv2.setMouseCallback(window_title, area_select, [window_title, img2, pt_1])
    cv2.waitKey()

    extracted_img = square_exchange(img2, pt_1.ptlist)

    # cv2.namedWindow(window_title)
    pt_2 = mp.PointLog(limit)
    cv2.setMouseCallback(window_title, area_select, [window_title, extracted_img, pt_2])
    cv2.waitKey()
    cv2.destroyAllWindows()

    print(pt_2.ptlist)

    x = pt_2.ptlist.transpose()[0].tolist()
    y = pt_2.ptlist.transpose()[1].tolist()

    (cx, cy, re) = CircleFitting(x, y)
    img_mask = np.zeros([extracted_img.shape[0], extracted_img.shape[1]], np.uint8)
    cv2.circle(img_mask, (cx, cy,), re, (255, 255, 255), -1)
    # cv2.imwrite('black_img.jpg', img_mask)
    mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

    # 切り抜き
    masked_upstate = cv2.bitwise_and(extracted_img, mask_rgb)

    # #
    # cv2.imshow("masked", masked_upstate)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    dst = masked_upstate

    tk = Trucker()
    tk.low_H = Green_lowH
    tk.low_S = Green_lowS
    tk.high_H = Green_highH

    cv2.namedWindow(tk.window_capture_name)
    cv2.namedWindow(tk.window_detection_name)
    cv2.createTrackbar(tk.low_H_name, tk.window_detection_name, tk.low_H, tk.max_value_H, tk.on_low_H_thresh_trackbar)
    cv2.createTrackbar(tk.high_H_name, tk.window_detection_name, tk.high_H, tk.max_value_H,
                       tk.on_high_H_thresh_trackbar)
    cv2.createTrackbar(tk.low_S_name, tk.window_detection_name, tk.low_S, tk.max_value, tk.on_low_S_thresh_trackbar)
    cv2.createTrackbar(tk.high_S_name, tk.window_detection_name, tk.high_S, tk.max_value, tk.on_high_S_thresh_trackbar)
    cv2.createTrackbar(tk.low_V_name, tk.window_detection_name, tk.low_V, tk.max_value, tk.on_low_V_thresh_trackbar)
    cv2.createTrackbar(tk.high_V_name, tk.window_detection_name, tk.high_V, tk.max_value, tk.on_high_V_thresh_trackbar)

    while (1):
        frame_HSV = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (tk.low_H, tk.low_S, tk.low_V), (tk.high_H, tk.high_S, tk.high_V))
        cv2.imshow(tk.window_capture_name, dst)
        cv2.imshow(tk.window_detection_name, frame_threshold)
        k = cv2.waitKey(1)
        # Escキーを押すと終了
        if k == 27:
            cv2.destroyAllWindows()
            break

    white_pixcels = cv2.countNonZero(frame_threshold)

    if __name__ == "__main__":
        frame_threshold = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2RGB)
        frame_threshold = cv2.bitwise_and(extracted_img, frame_threshold)
        cv2.imshow("Extracted view", frame_threshold)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return 5 * 5 * (white_pixcels / frame_threshold.size)


if __name__ == "__main__":
    area = calc_leaf_CR("test_CS.jpg")
    print(area)
