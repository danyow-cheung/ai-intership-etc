import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('x.jpeg',0)

'''
图像阈值
1.简单阈值
2.自适应阈值
3.Otsu's二值化
'''
'''-----------------简单阈值-----------------'''

# img = cv2.imread('x.jpeg',0)
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)


# titles = ['src','binary','binary_inv','trunc','tozero','tozero_inv']
# images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

'''-----------------自适应阈值-----------------
cv2.adaptiveTreshold(img,param1,param2,param3,param4,param5),其中
·param1是高于阈值判为的值
·param2选择阈值的方法
·param3是根据阈值判断像素点值的方法利用的是简单阈值分割的方法，
·param4是block_size(用来计算阈值的区域大小)
·param5是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。'''

#此处的0代表cv2.IMREAD_GRAYSCALE
# img1 = cv2.imread('x.jpeg',0)
# img = cv2.medianBlur(img1,5)
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# #11为block size,C为2
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# titles = ['src','global thresholding(v=127)','adaptive Mean Thresholding','adaptive Gaussian Thresholding']
# images = [img1,th1,th2,th3]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


'''-----------------Otsu's二值化-----------------'''
##Otsu's二值化
img = cv2.imread('x.jpeg',0)
#global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#Ostu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#核为(5,5),方差为0
blur = cv2.GaussianBlur(img,(5,5),0)
#阈值一定为0
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
images = [img,0,th1,
          img,0,th2,
          blur,0,th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
'Original Noisy Image','Histogram',"Otsu's Thresholding",
'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# 这里使用了 pyplot 中画直方图的方法，plt.hist, 要注意的是它的参数是一维数组
# 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用 flatten 方法
#ndarray.flat 1-D iterator over an array.
#ndarray.flatten 1-D array copy of the elements of an array in row-major order.
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
print (ret1,ret2,ret3)