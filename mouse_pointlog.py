import cv2
import numpy as np


class PointLog():
    def __init__(self, point_limit):
        self.ptlimit = point_limit
        self.ptlist = np.zeros((point_limit, 2), dtype=int)
        self.ptnum = 0

    def add(self, x, y):
        if self.ptnum < self.ptlimit:
            self.ptlist[self.ptnum, :] = [x, y]
            self.ptnum += 1
            return True
        return False
