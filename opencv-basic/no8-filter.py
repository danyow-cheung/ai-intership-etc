'''
图像滤波
·2d卷积
·平均滤波
·高斯滤波
·中值滤波
·双边滤波
'''
import cv2 
import numpy as np
from matplotlib import pyplot as plt 

'''-----------------·2d卷积-----------------'''
img = cv2.imread("x.jpeg")
kernel = np.ones((5,5),np.float32)/25

dst = cv2.filter2D(img,-1,kernel)

'''-----------------高斯滤波-----------------'''
#0 是指根据窗口大小（5,5）来计算高斯函数标准差
blur = cv2.GaussianBlur(img,(5,5),0)

'''-----------------中值滤波-----------------'''
mid_blur = cv2.medianBlur(img,5)

'''-----------------双边滤波-----------------'''
double_blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(121)
plt.imshow(img)
plt.title("src")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(dst)
plt.title("averaging")
plt.xticks([])
plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
# plt.show()

plt.subplot(122),plt.imshow(mid_blur),plt.title('mid-Blurred')
plt.xticks([]), plt.yticks([])
# plt.show()

plt.subplot(121),plt.imshow(double_blur),plt.title('double_blur-Blurred')
plt.xticks([]), plt.yticks([])
plt.show()