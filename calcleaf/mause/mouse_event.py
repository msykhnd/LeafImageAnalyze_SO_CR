import cv2
import numpy as np
from mause import mouse_pointlog as mp


def area_select(event, x, y, flag, params):
    _name, img, ptlist = params

    if event == cv2.EVENT_MOUSEMOVE:  # マウスが移動したときにx線とy線を更新する
        img2 = np.copy(img)
        h, w = img2.shape[0], img2.shape[1]
        cv2.line(img2, (x, 0), (x, h - 1), (255, 0, 0))
        cv2.line(img2, (0, y), (w - 1, y), (255, 0, 0))
        cv2.imshow(_name, img2)

    if event == cv2.EVENT_LBUTTONDOWN:
        if ptlist.add(x, y):
            print('[%d] ( %d, %d )' % (ptlist.ptnum - 1, x, y))
            cv2.circle(img, (x, y), 1, (0, 0, 255), 0)
            cv2.imshow(_name, img)
        else:
            print('All points have selected.  Press ESC-key.')

        if ptlist.ptnum == ptlist.ptlimit:
            # cv2.polylines(img,[ptlist.ptlist.reshape(-1,1,2)],True,(255,255,255))
            pass


if __name__ == "__main__":
    img = np.zeros((512, 512, 3), np.uint8)

    img_title = "mouse_event"
    cv2.namedWindow(img_title)
    limit = 4
    pt_class = mp.PointLog(limit)
    cv2.setMouseCallback(img_title, area_select, [img_title, img, pt_class])
    cv2.imshow(img_title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
