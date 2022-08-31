'''角点检测的FAST算法'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pic/pic.jpg')

fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,color=(255,0,0))

print ('Threshold:{}'.format(fast.getThreshold()))
print ('nonmaxSuppression:{}'.format(fast.getNonmaxSuppression()))
print ('neighborhood:{}'.format(fast.getType()))
print ('Total Keypoints with nonmaxSuppression:{}'.format(len(kp)))

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print ('Total Keypoints without nonmaxSuppression:{}'.format(len(kp)))
img3 = cv2.drawKeypoints(img,kp,None,color=(255,0,0))

plt.subplot(131),plt.imshow(img)
plt.title('src'),plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(img2)
plt.title('with NMS'),plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(img3)
plt.title('without NMS'),plt.xticks([]),plt.yticks([])
plt.show()