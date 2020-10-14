import cv2
import numpy as np
import math
from gui.trackers import Trucker
from mause import mouse_pointlog as mp
from mause.mouse_event import area_select
from utils.utils import imread

## Recomended Pram
## In future, use conf file

Green_lowH = 25
Green_highH = 80
Green_lowS = 80

def circle_fit(x, y):
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


class LeafImageProcessing:
    def __init__(self, file_name):
        self.file_name = file_name
        self.img = imread(self.file_name)
        self.resize_ratio = 0.2
        self.window_title = "Image"
        self.square_exchange_size = 500
        self.chanber_length = 0
        self.area_size = 0

    def square_exchange(self, vertax_pts):
        sq_size = self.square_exchange_size
        pts1 = np.float32(vertax_pts)
        pts2 = np.float32([[sq_size, sq_size], [sq_size, 0], [0, 0], [0, sq_size]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        self.img = cv2.warpPerspective(self.img, M, (sq_size, sq_size))
        return self.img

    def mouse_pointing(self, clk_limit=4):
        pt = mp.PointLog(clk_limit)
        cv2.setMouseCallback(self.window_title, area_select, [self.window_title, self.img, pt])
        cv2.waitKey()
        return pt.ptlist

    def resize_img(self):
        width = self.img.shape[1]
        height = self.img.shape[0]
        self.img = cv2.resize(self.img, (int(width * self.resize_ratio), int(height * self.resize_ratio)))

    def calc_leaf_CR(self):
        self.chanber_length = 5
        cv2.namedWindow(self.window_title)
        self.resize_img()
        ptlist = self.mouse_pointing()
        extracted_img = self.square_exchange(ptlist)
        ptlist = self.mouse_pointing()

        x = ptlist.transpose()[0].tolist()
        y = ptlist.transpose()[1].tolist()

        (cx, cy, re) = circle_fit(x, y)
        radius = math.sqrt(6 / math.pi)
        # radius ^ 2 * pi = 6 cm^2
        re = round(radius * self.img.shape[0] / 5)

        img_mask = np.zeros([extracted_img.shape[0], extracted_img.shape[1]], np.uint8)
        cv2.circle(img_mask, (cx, cy,), re, (255, 255, 255), -1)
        # cv2.imwrite('black_img.jpg', img_mask)
        mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

        # 切り抜き
        masked_upstate = cv2.bitwise_and(extracted_img, mask_rgb)

        dst = masked_upstate

        tk = Trucker()
        tk.low_H = Green_lowH
        tk.low_S = Green_lowS
        tk.high_H = Green_highH

        cv2.namedWindow(tk.window_capture_name)
        cv2.namedWindow(tk.window_detection_name)
        cv2.createTrackbar(tk.low_H_name, tk.window_detection_name, tk.low_H, tk.max_value_H,
                           tk.on_low_H_thresh_trackbar)
        cv2.createTrackbar(tk.high_H_name, tk.window_detection_name, tk.high_H, tk.max_value_H,
                           tk.on_high_H_thresh_trackbar)
        cv2.createTrackbar(tk.low_S_name, tk.window_detection_name, tk.low_S, tk.max_value, tk.on_low_S_thresh_trackbar)
        cv2.createTrackbar(tk.high_S_name, tk.window_detection_name, tk.high_S, tk.max_value,
                           tk.on_high_S_thresh_trackbar)
        cv2.createTrackbar(tk.low_V_name, tk.window_detection_name, tk.low_V, tk.max_value, tk.on_low_V_thresh_trackbar)
        cv2.createTrackbar(tk.high_V_name, tk.window_detection_name, tk.high_V, tk.max_value,
                           tk.on_high_V_thresh_trackbar)

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
        self.area_size = self.chanber_length ** 2 * (white_pixcels / frame_threshold.size)
        return frame_threshold

    def calc_leaf_SQ(self):
        self.chanber_length = 3
        cv2.namedWindow(self.window_title)
        self.resize_img()
        ptlist = self.mouse_pointing()
        extracted_img = self.square_exchange(ptlist)

        dst = extracted_img

        tk = Trucker()
        tk.low_H = Green_lowH
        tk.low_S = Green_lowS
        tk.high_H = Green_highH

        cv2.namedWindow(tk.window_capture_name)
        cv2.namedWindow(tk.window_detection_name)
        cv2.createTrackbar(tk.low_H_name, tk.window_detection_name, tk.low_H, tk.max_value_H,
                           tk.on_low_H_thresh_trackbar)
        cv2.createTrackbar(tk.high_H_name, tk.window_detection_name, tk.high_H, tk.max_value_H,
                           tk.on_high_H_thresh_trackbar)
        cv2.createTrackbar(tk.low_S_name, tk.window_detection_name, tk.low_S, tk.max_value, tk.on_low_S_thresh_trackbar)
        cv2.createTrackbar(tk.high_S_name, tk.window_detection_name, tk.high_S, tk.max_value,
                           tk.on_high_S_thresh_trackbar)
        cv2.createTrackbar(tk.low_V_name, tk.window_detection_name, tk.low_V, tk.max_value, tk.on_low_V_thresh_trackbar)
        cv2.createTrackbar(tk.high_V_name, tk.window_detection_name, tk.high_V, tk.max_value,
                           tk.on_high_V_thresh_trackbar)

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

        kernel = np.ones((5, 5), np.uint8)
        moph = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)
        moph = cv2.morphologyEx(moph, cv2.MORPH_CLOSE, kernel)

        cv2.imshow(tk.window_detection_name, moph)
        cv2.waitKey()
        cv2.destroyAllWindows()

        white_pixcels = cv2.countNonZero(moph)

        self.area_size = self.chanber_length ** 2 * (white_pixcels / frame_threshold.size)
        return moph


if __name__ == "__main__":
    # leaf_class = LeafImageProcessing("test_CS.jpg")
    # img = leaf_class.calc_leaf_CR()
    # cv2.imshow("Result",img)

    leaf_class = LeafImageProcessing("test_SQ.jpg")
    img = leaf_class.calc_leaf_SQ()
    cv2.imshow("Result", img)

    print(leaf_class.area_size)
