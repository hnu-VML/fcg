import numpy as np
import os
import cv2
from scipy.io import savemat
from scipy.io import loadmat
from scipy import signal

def fspecial(func_name, kernel_size, sigma):
    if func_name == 'gaussian':
        m = n = (kernel_size-1.)/2.
        y, x = np.ogrid[-m:m+1, -n:n+1]  # ogrid函数产生两个二维数组
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

def Gaussian_downsample(x, psf, s):
    y = np.zeros((x.shape[0], int(x.shape[1]/s), int(x.shape[2]/s)))
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y


def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F  # [3, 31]

# cave数据集的制作
path_s = '../CAVE/training_data/'
imglist_s = os.listdir(path_s)
save_s = './cave/train/'
training_size = 128
stride = 16
downsample_factor = 16
PSF = fspecial('gaussian', 7, 3)
R = create_F()  # 光谱波段采样矩阵

for i in range(len(imglist_s)):
    num = 0
    HRHSI = loadmat(path_s + imglist_s[i])['b']  # [H=512, W=512, C]
    HRHSI = np.transpose(HRHSI, (2, 0, 1))  # [H, W, C] -> [C, H, W]
    MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
    HSI = Gaussian_downsample(HRHSI, PSF, downsample_factor)

    for j in range(0, HRHSI.shape[1]-training_size+1, stride):
        for k in range(0, HRHSI.shape[2]-training_size+1, stride):
            temp_hrhs = HRHSI[:, j: j + training_size, k: k + training_size]
            temp_hrms = MSI[:, j: j + training_size, k: k + training_size]
            temp_lrhs = HSI[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            savemat(save_s+imglist_s[i][:-4]+'_'+str(num)+'.mat',{'hrhs':temp_hrhs, 'hrms':temp_hrms, 'lrhs':temp_lrhs})
            num=num+1
    print(i)

# Harvard数据集的制作
path_s = '../HARVARD/training_data/'
imglist_s = os.listdir(path_s)
save_s = './harvard/train/'
training_size = 128
stride = 40
downsample_factor = 16
PSF = fspecial('gaussian', 7, 3)
R = create_F()  # 光谱波段采样矩阵

for i in range(len(imglist_s)):
    num = 0
    HRHSI = loadmat(path_s + imglist_s[i])['ref']  # [H=1392, W=1040, C]
    HRHSI = HRHSI.astype(np.float64)/np.max(HRHSI)  # 归一化
    # savemat('harvard/test/'+imglist_s[i], {'b': HRHSI.astype(np.float32)})

    HRHSI = np.transpose(HRHSI, (2, 0, 1))  # [H, W, C] -> [C, H, W]
    MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
    HSI = Gaussian_downsample(HRHSI, PSF, downsample_factor)

    for j in range(0, HRHSI.shape[1]-training_size+1, stride):
        for k in range(0, HRHSI.shape[2]-training_size+1, stride):
            temp_hrhs = HRHSI[:, j: j + training_size, k: k + training_size]
            temp_hrms = MSI[:, j: j + training_size, k: k + training_size]
            temp_lrhs = HSI[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            savemat(save_s+imglist_s[i][:-4]+'_'+str(num)+'.mat',{'hrhs':temp_hrhs.astype(np.float32), 'hrms':temp_hrms.astype(np.float32), 'lrhs':temp_lrhs.astype(np.float32)})
            num=num+1
    print(i)
