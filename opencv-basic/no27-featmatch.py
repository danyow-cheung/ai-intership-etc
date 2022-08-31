'''特征匹配
1.Brute-Force匹配
2 FLANN匹配器
'''
'''-----------------1.Brute-Force匹配---------------
蛮力匹配器是很简单的。首先在第一幅图像中选取一个关键点然后依次与第二幅图
的每个关键点进行（描述符）距离测试，最后返回距离最近的关键点。
'''
from hashlib import algorithms_available
from tabnanny import check
import numpy as np
import cv2
from matplotlib import pyplot as plt
img2 = cv2.imread("pic/orange.jpg")
img1 = cv2.imread("pic/part-orange.jpg")

# orb = cv2.ORB_create()
# kp1,des1 =orb.detectAndCompute(img1,None)
# kp2,des2 =orb.detectAndCompute(img2,None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# matches = bf.match(des1,des2)

# matches = sorted(matches,key = lambda x:x.distance)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
# plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
# plt.show()


'''对SIFT描述符进行BF匹配和比值测试'''
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# good = []
# for m,n in matches:
#     if m.distance < 0.2 * n.distance:
#         good.append([m])

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
# plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
# plt.show()


'''-----------------1.FLANN匹配器---------------
FLANN 是快速最近邻搜索包(Fast_Library_for_Approximate_Nearest_Neighbors)的简称。
它是一个对大数据集和高维特征进行最近邻搜索的算法的集合，
而且这些算法都已经被优化过了。在面对大数据集时它的效果要好于BFMatcher。
'''

sift = cv2.SIFT_create()
kp1,des1 =sift.detectAndCompute(img1,None)
kp2,des2 =sift.detectAndCompute(img2,None)


FLAMM_INDEX_KDTREE=1
index_params = dict(algorithm=FLAMM_INDEX_KDTREE,trees=3)
seach_params = dict(checks=30)

flann = cv2.FlannBasedMatcher(index_params,seach_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.2*n.distance:
        matchesMask[i] = [1,0]       
        
draw_params = dict(matchColor = (0,255,0),
                  singlePointColor = (255,0,0),
                  matchesMask = matchesMask,
                  flags = cv2.DrawMatchesFlags_DEFAULT)
                  
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()
