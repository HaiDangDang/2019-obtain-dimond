import cv2
import numpy as np
import sys

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default
lowH = -2
lowS = -2
lowV = -2

highH = 2
highS = 2
highV = 2
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# mouse callback function

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + highH, pixel[1] + highS, pixel[2] + highV])
        lower =  np.array([pixel[0] + lowH, pixel[1] + lowS, pixel[2] + lowV])
        print(f'({lower[0]},{lower[1]},{lower[2]}),' + f'({upper[0]},{upper[1]},{upper[2]})')
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mask', 600, 600)
        result = image_src.copy()
        result[image_mask != 0] = (0, 0, 255)
        cv2.imshow("bgr",result)
        cv2.imshow("mask", image_mask)


global image_hsv, pixel # so we can use it in mouse callback
image_src = obs_array[100]
image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

cv2.namedWindow('bgr', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('bgr', pick_color)
cv2.resizeWindow('bgr', 600, 600)
cv2.imshow("bgr",image_src)

## NEW ##
cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('hsv', pick_color)
cv2.resizeWindow('hsv', 600, 600)

# now click into the hsv img , and look at values:

cv2.imshow("hsv",image_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_src = obs_array[100]
image_mask = np.zeros((64,64),dtype=np.uint8)
a = [(93,120,22),(113,140,72),(100,129,72),(104,133,76),(100,129,66),(104,133,70),
     (100,131,63),(104,135,67),(100,133,70),(104,137,74),(99,132,38),(103,136,42),
     (100,126,62),(104,130,66),(100,132,74),(104,136,78)]
for i in [0,2,4,6,8,10,12,14]:  
    image_mask1 = cv2.inRange(image_hsv,a[i],a[i+1])
    image_mask = image_mask1 + image_mask
    image_mask[image_mask > 255] = 255
result = image_src.copy()
result[image_mask != 0] = (0, 0, 255)
plt.imshow(result)
plt.show()
cnts, _= cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final_cnts = []
for c in cnts:
    if cv2.contourArea(c) > 4:
        final_cnts.append(c)
a = cv2.drawContours(image_src, final_cnts, -1,(0,0,255),1)
cv2.namedWindow('title', cv2.WINDOW_NORMAL)
cv2.resizeWindow('title', 600, 600)
cv2.imshow("title", a)
cv2.waitKey()
cv2.destroyAllWindows()
