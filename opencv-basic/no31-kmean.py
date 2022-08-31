'''K值聚类'''
'''仅有一个特征的数据'''
import numpy as np
import cv2 
from matplotlib import pyplot as plt 


# x = np.random.randint(25,100,25)
# y = np.random.randint(175,255,25)
# z = np.hstack((x,y))
# z = z.reshape((50,1))
# z = np.float32(z)


# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

# flags = cv2.KMEANS_RANDOM_CENTERS

# compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)

# A = z[labels==0]
# B = z[labels==1]

# plt.hist(A,256,[0,256],color='r')
# plt.hist(B,256,[0,256],color='b')

# plt.hist(centers,32,[0,256],color='y')
# plt.show()


'''二维特征聚类'''
# X = np.random.randint(25,50,(25,2))
# Y = np.random.randint(60,85,(25,2))
# Z = np.vstack((X,Y))

# Z = np.float32(Z)

# criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

# ret,label ,center = cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


# A = Z[label.ravel()==0]
# B = Z[label.ravel()==1]

# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c='r')
# plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

'''颜色量化'''
img = cv2.imread("pic/x.jpg")
Z = img.reshape((-1,3))

Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

K= 8
ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


center = np.uint8(center)
res = center[label.flatten()]
res2 =res.reshape((img.shape))

cv2.imshow("res2",res2)
cv2.waitKey(0)
cv2.destroyAllWindows()