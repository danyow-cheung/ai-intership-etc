'''
图像轮廓(轮廓绘制，轮廓特征(面积，周长，重心，边界框))、轮廓性质
'''


'''【轮廓】'''
import cv2 
# img = cv2.imread("pic/a.jpg")
# imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)

# contours ,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# img1 = cv2.drawContours(img,contours,-1,(0,0,0),3)
# cv2.imshow('img1',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''【轮廓特征】'''
# 矩
# cnt = contours[0]
# M = cv2.moments(cnt)
# print(M)
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# print(cx,cy)
# # 轮廓面积
# area = cv2.contourArea(cnt)
# print(area)
# #轮廓周长
# perimeter = cv2.arcLength(cnt,True)
# print (perimeter)


#轮廓近似
# img = cv2.imread('pic/a.jpg')
# imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# epsilon = 0.1*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# cv2.polylines(img,[approx],True,(0,0,255),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows

'''凸包
凸包与轮廓近似相似，但不同，虽然有些情况下它们给出的结果是一样的。
函数 cv2.convexHull() 可以用来检测一个曲线是否具有凸性缺陷，并能纠正缺陷
'''

# img2 = cv2.imread('pic/a.jpg')
# imggray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# hull = cv2.convexHull(cnt)
# # cv2.polylines(img2,[hull],True,(0,0,255),2)
# # cv2.imshow('img',img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # 凸性检测
# k  = cv2.isContourConvex(cnt)
# print(k)

# '''【直边界矩形】'''
# img3 = cv2.imread("pic/a.jpg")
# imggray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[1]

# x,y,w,h = cv2.boundingRect(cnt)
# img4 = cv2.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow('img',img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''[最小外接矩形]
这个边界矩形是面积最小的，因为它考虑了对象的旋转。
用到的函数为cv2.minAreaRect()。返回的是一个 Box2D 构，
其中包含矩形左上角角点的坐标（x，y），矩形的宽和高（w，h），以及旋转角度。
但是要绘制这个矩形需要矩形的 4 个角点，可以通过函数 cv2.boxPoints() 获得。
'''

import numpy as np
# img3 = cv2.imread("pic/a.jpg")
# imggray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[1]

# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.fillPoly(img3,[box],(0,0,255))
# cv2.imshow('img',img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''最小外接园'''
# img3 = cv2.imread("pic/a.jpg")
# imggray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[1]

# (x,y),radius = cv2.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# img4 = cv2.circle(img3,center,radius,(0,255,0),2)
# cv2.imshow('img',img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''【椭圆拟合】'''

# img3 = cv2.imread("pic/a.jpg")
# imggray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[1]

# ellipse = cv2.fitEllipse(cnt)
# img4 = cv2.ellipse(img3,ellipse,(0,255,0),2)
# cv2.imshow('img',img4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''【直线拟合】'''
# img3 = cv2.imread("pic/a.jpg")
# rows,cols = img3.shape[:2]
# imggray = cv2.cvtColor(img3,cv2.COLOR_RGB2GRAY)
# ret,thresh = cv2.threshold(imggray,0,255,cv2.THRESH_OTSU)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[1]

# [vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols -x)*vy/vx)+y)
# img = cv2.line(img3,(cols-1,righty),(0,lefty),(0,255,0),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#【宽高比】
# x,y,w,h = cv2.boundingRect(cnt)
# aspect_ratio = float(w)/h
# print(aspect_ratio)


# 轮廓面积与边界矩形面积的比
# area = cv2.contourArea(cnt)
# x,y,w,h= cv2.boundingRect(cnt)
# rect_area = w*h
# extent = float(area)/rect_area
# print(extent)

# 轮廓面积与凸包面积的比
# area = cv2.contourArea(cnt)
# hull = cv2.convexHull(cnt)
# hull_area = cv2.contourArea(hull)
# solidity = float(area)/hull_area
# print(solidity)

# 与轮廓面积相等的圆形直径
# area = cv2.contourArea(cnt)
# equi_diameter = np.sqrt(4*area/np.pi)
# print (equi_diameter)


# 方向
# (x,y),(Ma,ma) ,angle = cv2.fitEllipse(cnt)
# print ((x,y),(Ma,ma),angle)

# 轮廓的掩膜与像素点

# mask = np.zeros(img3.shape,np.uint8)
# mask = cv2.drawContours(mask,[cnt],0,255,-1)
# cv2.imshow('mask',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 最大值和最小值及它们的位置
# min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(img3,mask=mask)
# print (min_val,max_val,min_loc,max_loc)


# # 平均颜色及平均灰度
# mean_val = cv2.mean(img3,mask=mask)
# print (mean_val)


# # 极点
# leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
# rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
# topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
# bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
# print (leftmost,rightmost,topmost,bottommost)


'''凸缺陷
前面我们已经学习了轮廓的凸包，对象上的任何凹陷都被成为凸缺陷。
OpenCV 中有一个函数 cv.convexityDefect() 可以帮助我们找到凸缺陷。
如果要查找凸缺陷，在使用函数 cv2.convexHull 找凸包时，
参数returnPoints 一定要是 False。它会返回一个数组，其中每一行包含的值是 [起点，终点，最远的点，到最远点的近似距离。
'''
import cv2 
import numpy as np 
img = cv2.imread("pic/flyq.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[5]

# hull = cv2.convexHull(cnt,returnPoints=False)
# defects = cv2.convexityDefects(cnt,hull)
# for i in range(defects.shape[0]):
#     s,e,f,d = defects[i,0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])
#     cv2.line(img,start,end,[0,255,0],2)
#     cv2.circle(img,far,5,[0,0,255],-1)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Point Polygon Test

dist = cv2.pointPolygonTest(cnt,(50,50),True)
print (dist)

# 形状匹配
import cv2
import numpy as np
# 第一个图像
img = cv2.imread("pic/flyq.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[6]

# 第二个图像
img2 = cv2.imread('2.jpg',0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt2 = contours[0]
ret1 = cv2.matchShapes(cnt1,cnt2,1,0.0)
ret2 = cv2.matchShapes(cnt1,cnt1,1,0.0)
print (ret1,ret2)
