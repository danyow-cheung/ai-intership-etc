'''直方图均匀化、2D直方图和直方图反向投影'''
'''
【直方图均衡化】
如果一幅图片整体很亮，那所有的像素值应该都会很高。但是一副高质量的图像的像素值分布应该很广泛。
所以你应该把它的直方图做一个横向拉伸（如下图），这就是直方图均衡化要做的事情。通常情况下这种操作会改善图像的对比度。
'''
from turtle import color
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

img = cv2.imread("pic/demo.jpg",0)
hist ,bins = np.histogram(img.flatten(),256,[0,256])

# 计算累积分布图
cdf = hist.cumsum()
# cdf_normalized = cdf*hist.max()/cdf.max()

# plt.plot(cdf_normalized,color='b')
# plt.hist(img.flatten(),256,[0,256],color='r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'),loc='upper left')
# plt.show()

'''使用numpy进行灰度直方图均衡化'''
# 构建np掩膜数组，cdf为原数组，当数组元素为0时，掩盖（计算时被忽略）
# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min()) * 255/(cdf_m.max() - cdf_m.min())

# # 对掩盖的元素赋值，这里赋值为0
# cdf = np.ma.filled(cdf_m,0).astype("uint8")
# img2 = cdf[img]

# hist,bins = np.histogram(img2.flatten(),256,[0,256])

# 计算累积分布图
# cdf = hist.cumsum()
# cdf_normalized = cdf*hist.max()/cdf.max()

# plt.plot(cdf_normalized,color='b')
# plt.hist(img2.flatten(),256,[0,256],color='r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'),loc='upper left')
# plt.show()


'''opencv直方图均衡化'''

# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ))
# cv2.imshow('res',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#的确在进行完直方图均衡化之后，图片背景的对比度被改变了。
# 但是你再对比一下两幅图像中雕像的面图，由于太亮我们丢失了很多信息。
# 造成这种结果的根本原因在于这幅图像的直方图并不是集中在某一个区域。

# clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv2.imshow("clahe",cl1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''【2d直方图】
使用函数 cv2.calcHist() 来计算直方图既简单又方便。如果要绘制颜色直方图的话，我们首先需要将图像的颜色空间从 BGR 转换到 HSV。
'''
# opencv的2d直方图
# img = cv2.imread("pic/demo.jpg")
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# # hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
# # plt.imshow(hist,interpolation="nearest")
# # plt.show()

# # numpy的2d直方图

# hist,xbins,ybins = np.histogram2d(hsv[:,:,0].ravel(),hsv[:,:,1].ravel(),[180,256],[[0,180],[0,256]])
# plt.imshow(hist,interpolation="nearest")
# plt.show()

'''【直方图反向投影】
直方图反向投影它可以用来做图像分割，或者在图像中找寻我们感兴趣的部分
'''
# roi = cv2.imread("pic/part-orange.png")
# hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# target = cv2.imread("pic/orange.jpg")
# hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)


# M = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
# I = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
# R = M/I

# h,s,v = cv2.split(hsvt)
# B = R[h.ravel(),s.ravel()]
# B = np.minimum(B,1)
# B = B.reshape(hsvt.shape[:2])

# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# B = cv2.filter2D(B,-1,disc)
# B = np.uint8(B)
# cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
# ret,thresh = cv2.threshold(B,20,255,0)
# cv2.imshow('ss',thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# opencv方法
roi = cv2.imread('pic/part_orange.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
target = cv2.imread('pic/orange.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
roihist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
# 归一化：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
# cv2.NORM_MINMAX 对数组的所有值进行转化，使他们线性映射到最小值和最大值之间
#归一化之后的直方图便于显示，归一化之后就成了0到255之间的数了
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# 此处卷积可以把分散的点连在一起
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dst = cv2.filter2D(dst,-1,disc)
ret,thresh = cv2.threshold(dst,20,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
res = np.hstack((target,thresh,res))
cv2.imshow('1',res)
cv2.waitKey(0)