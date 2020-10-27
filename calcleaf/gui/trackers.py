import cv2
import numpy as np


class Tracker:
    def __init__(self):
        self.max_value = 255
        self.max_value_H = 360 // 2
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = self.max_value_H
        self.high_S = self.max_value
        self.high_V = self.max_value
        self.window_capture_name = 'Capture'
        self.window_detection_name = 'Detection'
        self.low_H_name = 'Low H'
        self.low_S_name = 'Low S'
        self.low_V_name = 'Low V'
        self.high_H_name = 'High H'
        self.high_S_name = 'High S'
        self.high_V_name = 'High V'

    def on_low_H_thresh_trackbar(self,val):

        self.low_H = val
        self.low_H = min(self.high_H - 1, self.low_H)
        cv2.setTrackbarPos(self.low_H_name, self.window_detection_name, self.low_H)

    def on_high_H_thresh_trackbar(self,val):

        self.high_H = val
        self.high_H = max(self.high_H, self.low_H + 1)
        cv2.setTrackbarPos(self.high_H_name, self.window_detection_name, self.high_H)

    def on_low_S_thresh_trackbar(self,val):

        self.low_S = val
        self.low_S = min(self.high_S - 1, self.low_S)
        cv2.setTrackbarPos(self.low_S_name, self.window_detection_name, self.low_S)

    def on_high_S_thresh_trackbar(self,val):

        self.high_S = val
        self.high_S = max(self.high_S, self.low_S + 1)
        cv2.setTrackbarPos(self.high_S_name, self.window_detection_name, self.high_S)

    def on_low_V_thresh_trackbar(self,val):

        self.low_V = val
        self.low_V = min(self.high_V - 1, self.low_V)
        cv2.setTrackbarPos(self.low_V_name, self.window_detection_name, self.low_V)

    def on_high_V_thresh_trackbar(self,val):
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V + 1)
        cv2.setTrackbarPos(self.high_V_name, self.window_detection_name, self.high_V)


def square_extract(img, vertax_pts, sq_size=900):
    pts1 = np.float32(vertax_pts)
    pts2 = np.float32([[sq_size, sq_size], [sq_size, 0], [0, 0], [0, sq_size]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    extract_img = cv2.warpPerspective(img, M, (900, 900))
    return extract_img


