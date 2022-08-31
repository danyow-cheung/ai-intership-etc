import cv2
import numpy as np
# img = cv2.imread('pic/x.jpeg',0)
# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# print (hist.shape)
# print (hist)
# img.ravel()#将图像转为一维数组，这里没有中括号
# img = np.array(img)
# hist,bins = np.histogram(img.ravel(),256,[0,256])
# print (hist.shape)
# print (len(bins))


# # 绘制直方图
from matplotlib import pyplot as plt 
# plt.hist(img.ravel(),256,[0,256])
# plt.show()

'''绘制三通道直方图'''

# img = cv2.imread('pic/x.jpeg')
# color = ('b','g','r')
# # 对一个列表或数组既要遍历索引又要遍历元素时，
# #使用内置enumerate函数会有更加直接和优美
# #enumerate会将数组或列表组成一个索引序列
# #是我们再获取索引内容的时候更加方便
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color=col)
#     plt.xlim([0,256])
# plt.show()

'''使用掩膜'''
img = cv2.imread('pic/x.jpeg',0)
mask = np.zeros(img.shape,dtype=np.uint8)
mask[100:200,100:200]=255
masked_img = cv2.bitwise_and(img,img,mask=mask)

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()