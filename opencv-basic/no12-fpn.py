'''
【高斯金字塔】

高斯金字塔的顶部是通过将底部图像中的连续的行和列去除得到的。
顶部图像中的每个像素值等于下一层图像中5 个像素的高斯加权平均值。
这样操作一次一个 M x N 的图像就变成了一个 M/2 x N/2 的图像。
所以这幅图像的面积就变为原来图像面积的四分之一。这被称为Octave。
连续进行这样的操作我们就会得到一个分辨率不断下降的图像金字塔。
'''
# import cv2 
# img = cv2.imread("pic/x.jpeg")
# lower_reso = cv2.pyrDown(img)
# higer_reso = cv2.pyrUp(lower_reso)

# cv2.imshow('src',img)
# cv2.imshow('l',lower_reso)
# cv2.imshow('h',higer_reso)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
【拉普拉斯金字塔】
使用金字塔进行图像融合
图像金字塔的一个应用是图像融合。
例如，在图像缝合中，你需要将两幅图叠在一起，
但是由于连接区域图像像素的不连续性，整幅图的效果看起来会很差。
这时图像金字塔就可以排上用场了，他可以帮你实现无缝连接。
这里的一个经典案例就是将两个水果融合成一个，看看下图也许你就明白我在讲什么了。

'''
import numpy as np 
import cv2 
a = cv2.imread("pic/apple.jpg")
a = cv2.resize(a,(256,256))

o = cv2.imread("pic/orange.jpg")
o = cv2.resize(o,(256,256))


# 生成a的高斯金字塔
G = a.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
#生成o的高斯金字塔
G = o.copy()
gpO = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpO.append(G)


#生成A的拉普拉斯金字塔
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L= cv2.subtract(gpA[i-1],GE)
    lpA.append(L)
#生成O的拉普拉斯金字塔
lpO = [gpO[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpO[i])
    L= cv2.subtract(gpO[i-1],GE)
    lpO.append(L)

#现在在每个级别中添加左右两半图像
LS = []
for la, lb in zip(lpA, lpO):
    rows,cols,dpt = la.shape
    # 图像拼接
    ls = np.hstack((la[:,0:int(cols/2)],lb[:,int(cols/2):]))
    LS.append(ls)
# 现在重建
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

#图像与直接连接的每一半
real1 = np.hstack((a, o))
real = np.hstack((a[:, 0:int(cols/2)], o[:, int(cols/2):]))
cv2.imshow('real1', real1)
cv2.imshow('real', real)
cv2.imshow('ls_', ls_)
cv2.waitKey(0)
cv2.destroyAllWindows()
