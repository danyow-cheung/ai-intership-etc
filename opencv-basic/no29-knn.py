'''K 近邻(k-Nearest Neighbour'''
from re import S
import cv2 
import numpy as np
from matplotlib import pyplot as plt 

trainData = np.random.randint(0,100,(25,2)).astype(np.float32)


responses = np.random.randint(0,2,(25,1)).astype(np.float32)


red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

newcomers = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomers[:,0],newcomers[:,1],80,'g','o')


knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret,results,neighbours,dist = knn.findNearest(newcomers,3)

print ('result:{}'.format(results),'\n')
print ('neightbours:{}'.format(neighbours))
print ('distance:{}'.format(dist))

plt.show()