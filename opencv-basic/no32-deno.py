'''-计算摄影学图像去噪

cv2.fastNlMeansDenoising() 使用对象为灰度图。
cv2.fastNlMeansDenoisingColored() 使用对象为彩色图。
cv2.fastNlMeansDenoisingMulti() 适用于短时间的图像序列（灰度图像）。
cv2.fastNlMeansDenoisingColoredMulti() 适用于短时间的图像序列（彩色图像）。
'''
import cv2 
from matplotlib import pyplot as plt 

img = cv2.imread("pic/part-orange.jpg")
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()