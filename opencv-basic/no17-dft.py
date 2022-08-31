'''傅里叶变换经常被用来分析不同滤波器的频率特性。
我们可以使用2D离散傅里叶变换 (DFT) 分析图像的频域特性。
实现 DFT 的一个快速算法被称为快速傅里叶变换（FFT）。
边界和噪声是图像中的高频分量（注意这里的高频是指变化非常快，而非出现的次数多）
。如果没有如此大的幅度变化我们称之为低频分量。
'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

# img = cv2.imread("pic/x.jpeg",0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)

# magnitude_spe = 20 * np.log(np.abs(fshift))
# plt.subplot(121),plt.imshow(img,cmap='gray')
# plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spe,cmap='gray')
# plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
# plt.show()

'''现在我们可以进行频域变换了，我们就可以在频域对图像进行一些操作了，
例如高通滤波和重建图像（DFT 的逆变换）。
比如我们可以使用一个60x60 的矩形窗口对图像进行掩模操作从而去除低频分量
。然后再使用函数np.fffft.ifffftshift() 进行逆平移操作，
所以现在直流分量又回到左上角了，左后使用函数 np.ifffft2() 进行 FFT 逆变换。
同样又得到一堆复杂的数字，我们可以对他们取绝对值
'''
# rows,cols = img.shape
# crow,ccol = int(rows/2),int(cols/2)

# fshift [crow-30:crow+30,ccol-30:ccol+30] =0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)

# 取绝对值
# img_back = np.abs(img_back)
# plt.subplot(131),plt.imshow(img,cmap='gray')
# plt.title("input image"),plt.xticks([]),plt.yticks([])
# plt.subplot(132),plt.imshow(img_back,cmap='gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.show()




'''____________________________________opencv中的傅立叶变换________________________________________'''
img = cv2.imread("pic/a.jpg",0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft[:,:,1]))
# plt.subplot(121),plt.imshow(img,cmap='gray')
# plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray')
# plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
# plt.show()

# 做逆dft
# 其实就是对图像进行模糊操作。首先我们需要构建一个掩模，
# 与低频区域对应的地方设置为 1, 与高频区域对应的地方设置为 0。
rows,cols = img.shape
crow,ccol = int(rows/2),int(cols/2)

mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30]=1
fshift =dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()


# 各种滤波器的高迪通属性
# simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))
# creating a guassian filter
x = cv2.getGaussianKernel(5,10)
#x.T 为矩阵转置
gaussian = x*x.T
# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                    [-10,0,10],
                    [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                    [0, 0, 0],
                    [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])
filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()