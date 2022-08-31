'''使用特征匹配和单应性查找对象'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img2 = cv2.imread("pic/orange.jpg")
img1 = cv2.imread("pic/part-orange.jpg")

sift = cv2.SIFT_create()
# 找到关键点
kp1,des1 =sift.detectAndCompute(img1,None)
kp2,des2 =sift.detectAndCompute(img2,None)


FLAMM_INDEX_KDTREE=0
index_params = dict(algorithm=FLAMM_INDEX_KDTREE,trees=5)
seach_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,seach_params)
matches = flann.knnMatch(des1,des2,k=2)
# 存储所有的好的匹配
good =[]
for m,n in matches:
    if m.distance <0.2 * n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)



# 第三个参数 Method used to computed a homography matrix. The following methods are possible:
#0 - a regular method using all the points
#CV_RANSAC - RANSAC-based robust method
#CV_LMEDS - Least-Median robust method
# 第四个参数取值范围在 1 到 10，ᲁ绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差
# 超过误差就认为是 outlier
# 返回值中 M 为变换矩阵。

    M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5,0)
    matchesMask = mask.ravel().tolist()

    h,w,_ = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)    
    dst = cv2.perspectiveTransform(pts,M)

    cv2.polylines(img2,[np.int32(dst)],True,255,10,cv2.LINE_AA)

else:
    print("no enough matches are found -{}{}".format(len(good),MIN_MATCH_COUNT))
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()
