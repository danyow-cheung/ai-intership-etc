
'''尺度不变特征变换（SIFT），
这个算法可以帮助我们提取图像中的关键点并计算它们的描述符。'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pic/pic.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 创建sift检测器
sift = cv2.SIFT_create()

kp,res = sift.detectAndCompute(gray,None)

img = cv2.drawKeypoints(img,outImage=img,keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)
plt.show()