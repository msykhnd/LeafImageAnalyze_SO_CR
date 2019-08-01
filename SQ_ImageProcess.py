import cv2
import numpy as np
import mouse_pointlog as mp
from mouse_event import area_select
from trackers import Trucker

Green_lowH = 25
Green_highH = 80
Green_lowS = 60


def square_exchange(img, vertax_pts, sq_size=600):
    pts1 = np.float32(vertax_pts)
    pts2 = np.float32([[sq_size, sq_size], [sq_size, 0], [0, 0], [0, sq_size]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    extract_img = cv2.warpPerspective(img, M, (sq_size, sq_size))
    return extract_img


def calc_leaf_SQ(File_name, resized_ratio=0.2):
    img = cv2.imread(File_name, cv2.IMREAD_UNCHANGED)

    img2 = cv2.resize(img, dsize=None, fx=resized_ratio, fy=resized_ratio)
    window_title = "Area_Select"
    cv2.namedWindow(window_title)

    limit = 4
    pt_class = mp.PointLog(limit)
    cv2.setMouseCallback(window_title, area_select, [window_title, img2, pt_class])
    cv2.waitKey()
    cv2.destroyAllWindows()

    dst = square_exchange(img2, pt_class.ptlist,sq_size=600)
    dst = cv2.medianBlur(dst, 5)

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
        cv2.imshow(window_title, dst)
        cv2.imshow(tk.window_detection_name, frame_threshold)
        k = cv2.waitKey(1)
        # Escキーを押すと終了
        if k == 10 or k == 27:
            break
    kernel = np.ones((5, 5), np.uint8)
    moph = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)
    moph = cv2.morphologyEx(moph, cv2.MORPH_CLOSE, kernel)

    cv2.imshow(tk.window_detection_name, moph)

    white_pixcels = cv2.countNonZero(moph)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return 3 * 3 * (white_pixcels / frame_threshold.size)


if __name__ == "__main__":
    area = calc_leaf_SQ("test_SQ.jpg")
    print(area)
