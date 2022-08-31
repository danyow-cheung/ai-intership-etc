'''
[模板匹配]
单目标匹配：
多对象的模板匹配：
'''

'''单目标匹配'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

# img = cv2.imread('pic/orange.jpg', 0)
# img2 = img.copy()
# template = cv2.imread('pic/part-orange.png', 0)
# w, h = template.shape[::-1]

# # 六种比较方法
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)

#     res = cv2.matchTemplate(img,template,method)
#     min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

#     # 使用不同的方法比较，对结果的解释不同
#     if method is [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0]+w,top_left[1]+h)

#     cv2.rectangle(img,top_left,bottom_right,255,2)
#     plt.subplot(121), plt.imshow(res, cmap='gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122), plt.imshow(img, cmap='gray')
#     plt.title('Detect Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()

'''多对象的模版匹配'''
img = cv2.imread('pic/orange.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


gray =cv2.imread('pic/orange.jpg', 0)
template = cv2.imread('pic/part-orange.png', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
print(res[0])
threshold = 0.8
loc =  np.where(res>= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)

plt.subplot(131), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(template, cmap='gray')
plt.title('Roi'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img, cmap='gray')
plt.title('Detect Point'), plt.xticks([]), plt.yticks([])
plt.show()