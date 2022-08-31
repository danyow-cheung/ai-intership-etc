'''分水岭算法图像分割'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread("pic/circle.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 大津阈值分割并将前后背景颜色反转
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# plt.imshow(thresh,cmap='gray')
# plt.show()


# 得到前后景
kernel = np.ones((3, 3), dtype=np.uint8)
# 开运算去除白噪声
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# 膨胀操作得到背景
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
# 距离变换的基本含义是计算一个图像中非零像素点到最近的零像素点的距离，也就是到零像素点的最短距离
# 其最常见的距离变换算法就是通过连续的腐蚀操作来实现，腐蚀操作的停止条件是所有前景像素都被完全
# 腐蚀。这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心骨架像素点的
# 距离。根据各个像素点的距离值，设置为不同的灰度值。这样就完成了二值图像的距离变换
# cv2.distanceTransform(src, distanceType, maskSize)
# 第二个参数 0,1,2 分别表示 CV_DIST_L1, CV_DIST_L2 , CV_DIST_C
dist_transform = cv2.distanceTransform(opening, 1, 5)
print(np.unique(dist_transform))
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# plt.subplot(141), plt.imshow(sure_bg, cmap='gray')
# plt.title('bg'), plt.xticks([]), plt.yticks([])
# plt.subplot(142), plt.imshow(sure_fg, cmap='gray')
# plt.title('fg'), plt.xticks([]), plt.yticks([])
# plt.subplot(143), plt.imshow(dist_transform, cmap='gray')
# plt.title('dist_transfrom'), plt.xticks([]), plt.yticks([])
# plt.subplot(144), plt.imshow(unknown, cmap='gray')
# plt.title('unknown'), plt.xticks([]), plt.yticks([])

# plt.show()
ret, markers1 = cv2.connectedComponents(sure_fg)
markers = markers1 + 1
markers[unknown == 255] = 0
print (np.unique(markers))
# plt.imshow(markers,cmap='jet')
# plt.title('markers'),plt.xticks([]),plt.yticks([])
# plt.show()

markers3 = cv2.watershed(img, markers)
img[markers3 == -1] = [255, 0, 0]

plt.subplot(121), plt.imshow(markers3, cmap='jet')
plt.title('mark'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img)
plt.title('res'), plt.xticks([]), plt.yticks([])

plt.show()