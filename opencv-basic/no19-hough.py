'''Hough直线变换和Hough圆环变换'''

# hough直线变换
import cv2 
import numpy as np

# img = cv2.imread("pic/chess.jpg")
# gray = cv2.imread("pic/chess.jpg",0)

# edges = cv2.Canny(gray,50,150,apertureSize=3)
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for i in range(len(lines)):
#     for rho ,theta in lines[i]:
        
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0+1000*(-b))
#         y1 = int(y0+1000*(a))
#         x2 = int(x0-1000*(-b))
#         y2 = int(y0-1000*(a))
        
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        
# cv2.imshow('111',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# robabilistic_Hough_Transform 是对霍夫变换的一种优化。
# 它不会对每一个点都进行计算，而是从一幅图像中随机选取
# （是不是也可以使用图像金字塔呢？）一个点集进行计算，
# 对于直线检测来说这已经足够了。但是使用这种变换我们必须要降低阈值
# （总的点数都少了，阈值肯定也要小呀！）。

# img = cv2.imread("pic/chess.jpg")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize=3)

# minLineLenghth = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLenghth,maxLineGap)
# for i in range(len(lines)):
#     for x1,y1,x2,y2 in lines[i]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow("222",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''圆形变换'''

img = cv2.imread("pic/circle.jpg",0)
gray = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=50,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) 
    
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

