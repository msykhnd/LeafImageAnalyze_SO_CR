import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)


# mouse callback functions
refPt = []
cropping = False
# Double click pointing & put circle
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


# Click & Corp function
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img)


# 4 Pointing & plot squarer

def point_vertax_and_plot_square(event, x, y, flags, param):
    global refPt
    point_count = 0
    if event == cv2.EVENT_LBUTTONDBLCLK:
        refPt.append((x, y))
        point_count += 1
        print(point_count)


# Create a black image, a window and bind the function to window
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', point_vertax_and_plot_square)

while (1):
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        break

img = cv2.polylines(img, [np.array(refPt, np.int32).reshape((-1, 1, 2))], True, (0, 255, 255))
cv2.imshow('image3', img)
print(refPt)
cv2.destroyAllWindows()
print("Clear")
