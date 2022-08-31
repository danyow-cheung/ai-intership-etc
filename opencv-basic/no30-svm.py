'''使用SVM进行手写数据OCR
在 kNN 中我们直接使用像素的灰度值作为特征向量。
这次我们要使用方向梯度直方图Histogram of Oriented Gradients （HOG）作为特征向量。'''

import cv2 
import numpy as np
from matplotlib import pyplot as plt 

SZ=20
bin_n = 16 
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


# 使用方向梯度直方图作为特征向量
def deskew(img):
    # 对一个图像进行抗扭斜处理
    m = cv2.moments(img)

    if abs(m['mu02'])<1e-2:
        return img.copy()
    
    skew = m['mu11']/m['mu02']
    M = np.float32([[1,skew,-0.5*SZ*skew],[0,1,0]])

# 图像的平移，参数:输入图像、变换矩阵、变换后的大小
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img



def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


img =cv2.imread('pic/digits.png',0)
if img is None:
    raise Exception("we need the .png file")


cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]


deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]

trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:np.newaxis]


svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)


svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]


mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
