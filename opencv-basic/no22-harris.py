'''harris角点检测'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread("pic/chess.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 输入图像必须是float32，最后一个参数在0.04-0.05
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)

# 阈值
# img[dst>0.01 *dst.max()]==[0,0,255]
# cv2.imshow('dst',img)
# if cv2.waitKey(0)==27:
#     cv2.destroyAllWindows()
''' 亚像素级精确度的角点'''
ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)


ret,labels ,stats,centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER,100,0.001)

corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

res = np.hstack((centroids,corners))

res = np.int0(res)
img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,2555,0]
cv2.imshow('111',img)
cv2.waitKey(0)