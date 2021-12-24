import cv2
import numpy as np

ORD_ESCAPE = 0x1b #ESC
BackendError = type('BackendError', (Exception,), {})

# cv2.imread が日本語ファイル，ディレクトリを読み込めないため
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def _is_visible(winname):
    try:
        ret = cv2.getWindowProperty(
            winname, cv2.WND_PROP_VISIBLE
        )
        if ret == -1:
            raise BackendError('Use Qt as backend to check whether window is visible or not.')
        return bool(ret)

    except cv2.error:
        return False

def closeable_imshow(winname, img, *, break_key=ORD_ESCAPE):
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKey(10)

        if key == break_key:
            break
        if not _is_visible(winname):
            break

    cv2.destroyWindow(winname)