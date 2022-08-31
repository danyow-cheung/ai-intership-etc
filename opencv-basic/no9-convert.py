'''
——————————————形态学转换——————————————
腐蚀
膨胀
开闭运算
礼帽
黑帽
形态学梯度运算
—————————————————————————————————————
'''


'''
—————————————————腐蚀————————————————————
就像土壤侵蚀一样，这个操作会把前景物体的边界腐蚀掉(但是前景仍然是白色)。
根据卷积核的大小靠近前景的所有像素都会被腐蚀掉(变为 0)，所以前景物体会变小，
整幅图像的白色区域会减少。这对于去除白噪声很有用，也可以用来断开两个连在一块的物体等。
'''
import cv2 
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread("pic/a.jpg")
ret,binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# kernel_1 = np.ones((5,5),dtype=np.uint8)
# erosion1 = cv2.erode(binary,kernel_1,iterations=1)
# plt.subplot(121),plt.imshow(binary),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(erosion1),plt.title('Eroded1')
# plt.xticks([]), plt.yticks([])
# plt.show()
'''
—————————————————膨胀————————————————————
这个操作会增加图像中的白色区域（前景）。一般在去噪声时先用腐蚀再用膨胀。
因为腐蚀在去掉白噪声的同时，也会使前景对象变小。所以我们再对他进行膨胀。
这时噪声已经被去除了，不会再回来了，
但是前景还在并会增加。膨胀也可以用来连接两个分开的物体。
'''
kernel_1 = np.ones((5,5),dtype=np.uint8)
# erosion1 = cv2.dilate(binary,kernel_1,iterations=1)

# plt.subplot(121),plt.imshow(binary),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(erosion1),plt.title('erosion1')
# plt.xticks([]), plt.yticks([])
# plt.show()

'''
—————————————————开运算————————————————————
先进性腐蚀再进行膨胀就叫做开运算。腐蚀在去掉白噪声的同时，也会使前景对象变小。
所以我们再对他进行膨胀。这时噪声已经被去除了，不会再回来了，但是前景还在并会增加
'''
# opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel_1)

# plt.subplot(121),plt.imshow(binary),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(opening),plt.title('opening')
# plt.xticks([]), plt.yticks([])
# plt.show()


'''
—————————————————闭运算————————————————————
先进性腐蚀再进行膨胀就叫做开运算。腐蚀在去掉白噪声的同时，也会使前景对象变小。
所以我们再对他进行膨胀。这时噪声已经被去除了，不会再回来了，但是前景还在并会增加
'''

# closing = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel_1)

# plt.subplot(121),plt.imshow(binary),plt.title('Original')
# plt.xticks([]), plt.yticks([])


# plt.subplot(122),plt.imshow(closing),plt.title('closing')
# plt.xticks([]), plt.yticks([])
# plt.show()


'''
—————————————————形态学梯度————————————————————
一幅图像膨胀与腐蚀的差，结果看上去就像前景物体的轮廓。
'''

# gradient = cv2.morphologyEx(binary,cv2.MORPH_GRADIENT,kernel_1)

# plt.subplot(121),plt.imshow(binary),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(gradient),plt.title('Gradient')
# plt.xticks([]), plt.yticks([])
# plt.show()

'''
—————————————————礼帽————————————————————
礼帽 = 原图像-开运算图像，得到的是噪声图像。
'''
kernel_2 = np.ones((9,9),dtype=np.uint8)
# tophat = cv2.morphologyEx(binary,cv2.MORPH_TOPHAT,kernel_2)


# plt.subplot(121),plt.imshow(binary),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(tophat),plt.title('tophat')
# plt.xticks([]), plt.yticks([])
# plt.show()

'''
—————————————————黑帽————————————————————
黑帽 = 闭运算图片 - 原始图片，得到图像内部的小孔，或者背景色中的小黑点。
'''

blackhat = cv2.morphologyEx(binary,cv2.MORPH_BLACKHAT,kernel_2)


plt.subplot(121),plt.imshow(binary),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blackhat),plt.title('blackhat')
plt.xticks([]), plt.yticks([])
plt.show()

