import cv2
sky_mask1 = cv2.inRange(image_hsv, (0, 105, 215), (20, 125, 295))
sky_mask2 = cv2.inRange(image_hsv, (1, 69, 215), (21, 89, 295))
sky_mask3 = cv2.inRange(image_hsv, (0, 88, 215), (20, 108, 295))
sky_mask = sky_mask1 + sky_mask2 + sky_mask3
cv2.namedWindow('sky_mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('sky_mask', 600, 600)
cv2.imshow("sky_mask", sky_mask)



water_mask1 = cv2.inRange(image_hsv, (-7, 193, 130), (13, 213, 210))
water_mask2 = cv2.inRange(image_hsv, (-7, 170, 144), (13, 190, 224))
water_mask3 = cv2.inRange(image_hsv, (-8, 161, 142), (12, 181, 222))
water_mask4 = cv2.inRange(image_hsv, (-7, 183, 134), (13, 203, 214))
water_mask = water_mask1 + water_mask2 + water_mask3 + water_mask4
cv2.namedWindow('water_mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('water_mask', 600, 600)
cv2.imshow("water_mask", water_mask)


# DIRT
a = [(90,127,72),(100,137,82),(97,122,89),(117,142,119),(97,118,124),(117,138,154),
     (97,87,80),(117,107,110),(96,119,123),(116,139,153),(96,119,108),(116,139,138),
     (106,136,71),(110,140,81),(103,128,78),(113,138,98),(97,105,90),(117,125,140),
     (98,93,89),(118,113,139),(98,63,77),(118,83,127)]
for i in [0,2,4,6,8,10,12,14,16,18,20]:

#TRee
a = [(93, 120, 22), (113, 140, 72), (100, 129, 72), (104, 133, 76), (100, 129, 66), (104, 133, 70),
     (100, 131, 63), (104, 135, 67), (100, 133, 70), (104, 137, 74), (99, 132, 38), (103, 136, 42),
     (100, 126, 62), (104, 130, 66), (100, 132, 74), (104, 136, 78)]
for i in [0, 2, 4, 6, 8, 10, 12, 14]:

import numpy as np
import cv2

cap = cv2.VideoCapture('recording.mp4')
import tensorflow as tf
while (cap.isOpened()):
     ret, frame = cap.read()
     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     image_hsv[..., 1] = image_hsv[..., 0] * 1.2
     image_mask = np.zeros((64, 64), dtype=np.uint8)
     a = [(0, 105, 215), (20, 125, 295), (1, 69, 215), (21, 89, 295), (0, 88, 215),
          (20, 108, 295)]

     for i in [0, 2, 4]:
          image_mask1 = cv2.inRange(image_hsv, a[i], a[i + 1])
          image_mask = image_mask1 + image_mask
          image_mask[image_mask > 255] = 255
     cnts, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     final_cnts = []
     for c in cnts:
          if cv2.contourArea(c) > 4:
               final_cnts.append(c)
     a = cv2.drawContours(frame, final_cnts, -1, (0, 0, 255), 1)

     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
     cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
     cv2.resizeWindow('image', 600, 600)
     cv2.resizeWindow('mask', 600, 600)

     cv2.imshow("mask", image_hsv)
     cv2.imshow("image", frame)
     if cv2.waitKey(50) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()

image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
image_hsv[...,0] = image_hsv[...,0]*1.2
img_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
plt.imshow(image_hsv)
plt.show()
