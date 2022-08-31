'''grabcut前景分割'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('pic/apple.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mask = np.zeros(img.shape[:2],np.uint8)


bgModel = np.zeros((1,65),np.float64)
fgModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
cv2.grabCut(img,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
img_ = img * mask2[:, :, np.newaxis]
plt.subplot(121), plt.imshow(img)
plt.title('img'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_)
plt.title('img_'), plt.xticks([]), plt.yticks([])
plt.show()