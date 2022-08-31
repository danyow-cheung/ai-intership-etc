'''在 OpenCV 中只需要一个函数：cv2.Canny()，
就可以完成以上几步。让我们看如何使用这个函数。
这个函数的第一个参数是输入图像。
第二和第三个分别是 minVal 和 maxVal。
第三个参数设置用来计算图像梯度的 Sobel卷积核的大小，默认值为 3。

最后一个参数是 L2gradient，它可以用来设定求梯度大小的方程。
如果设为 True，就会使用我们上面提到过的方程，否则使用方程：
Edgee Gradient(G) = |Gx2| + |Gy2| 代替，默认值为 False。
'''

import cv2 
img = cv2.imread("pic/x.jpeg")
cv2.namedWindow("canny")
# 定义回调函数
def nothing(x):
    pass

# 创建滑动条，分别控制threshold1，threshold2
cv2.createTrackbar("threshold1",'canny',50,400,nothing)
cv2.createTrackbar("threshold2",'canny',100,400,nothing)
while(1):
    # 返回滑动条所在位置的数
    threshold1 = cv2.getTrackbarPos('threshold1','canny')
    threshold2 = cv2.getTrackbarPos('threshold2','canny')

    # canny边缘检测
    edge = cv2.Canny(img,threshold1,threshold2)

    # 显示图片
    cv2.imshow('origial',img)
    cv2.imshow('canny',edge)

    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()

    