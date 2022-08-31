'''
        图像梯度
1.sobel
2.scharr
3.laplacian
'''


'''
【Sobel算子、Scharr算子】
Sobel(一阶导数)算子是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好。
你可以设定求导的方向（xorder 或 yorder）。
还可以设定使用的卷积核的大小（ksize）。
如果 ksize=-1，会使用 3x3 的 Scharr 滤波器，它的的效果要比 3x3 的 Sobel 滤波器好
（而且速度相同，所以在使用 3x3 滤波器时应该尽量使用 Scharr 滤波器）。
'''
import cv2 
import numpy as np
from matplotlib import pyplot as plt 

box = np.zeros((200,200),dtype=np.uint8)
box[30:180,30:180]=255
# img = box.copy()

# # cv2.CV_64F输出图像的深度（数据类型），可以使用-1，与原图保持一致np.unit8
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# # 参数1，22为只在x方向求一阶导数，最大可以求2阶导数
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

# # 参数1，22为只在y方向求一阶导数，最大可以求2阶导数
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# cv2.imshow("src",img)
# cv2.imshow('lpls',laplacian)
# cv2.imshow('s_x',sobelx)
# cv2.imshow('s_y',sobely)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
如果原图像的深度是np.int8 时，所有的负值都会被截断变成 0，换句话说就是把把边界丢失掉。
所以如果这两种边界你都想检测到，最好的的办法就是将输出的数据类型设置的更高，
比如 cv2.CV_16S，cv2.CV_64F 等。取绝对值然后再把它转回到 cv2.CV_8U。下
面的示例演示了输出图片的深度不同造成的不同效果。

'''
sobelx8u = cv2.Sobel(box,cv2.CV_8U,1,0,ksize=5)
# 将参数设为1
sobelx64f = cv2.Sobel(box,cv2.CV_64F,1,0,ksize=5)

abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(1,3,1),plt.imshow(box,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()