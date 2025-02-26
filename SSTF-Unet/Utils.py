from scipy.io import loadmat
import numpy as np
import torch
import os
from scipy import signal
import torch.utils.data as data


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


class HSI_MSI_Data(data.Dataset):
    def __init__(self, path_s):
        self.path_s = path_s
        self.imglist_s = os.listdir(path_s)

    def __getitem__(self, index):
        mat = loadmat(self.path_s + self.imglist_s[index])
        hrhs = mat['hrhs']  # [C,H,W]
        hrms = mat['hrms']  # [C,H,W]
        lrhs = mat['lrhs']  # [C,H,W]

        hrhs = torch.from_numpy(hrhs.astype(np.float32))
        hrms = torch.from_numpy(hrms.astype(np.float32))
        lrhs = torch.from_numpy(lrhs.astype(np.float32))

        return hrhs, hrms, lrhs

    def __len__(self):
        return len(self.imglist_s)

# class HSI_MSI_Data(data.Dataset):
#     def __init__(self, path, R, PSF, training_size, stride, downsample_factor, num):
#         imglist = os.listdir(path)
#         train_hrhs = []
#         train_hrms = []
#         train_lrhs = []
#         for i in range(num):
#             img = loadmat(path + imglist[i])  # img为一个字典
#             img1 = img['b']  # 通过key引用字典的value
#             HRHSI = np.transpose(img1, (2, 0, 1))  # [H, W, C] -> [C, H, W]
#             MSI = np.tensordot(R,  HRHSI, axes=([1], [0]))
#             HSI = Gaussian_downsample(HRHSI, PSF, downsample_factor)
#             for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
#                 for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
#                     temp_hrhs = HRHSI[:, j: j + training_size, k: k + training_size]
#                     temp_hrms = MSI[:, j: j + training_size, k: k + training_size]
#                     temp_lrhs = HSI[:, int(j / downsample_factor):int((j + training_size ) /downsample_factor), int( k /downsample_factor):int(( k +training_size ) /downsample_factor)]
#                     train_hrhs.append(temp_hrhs)
#                     train_hrms.append(temp_hrms)
#                     train_lrhs.append(temp_lrhs)
#
#         train_hrhs = torch.Tensor(np.array(train_hrhs))
#         train_lrhs = torch.Tensor(np.array(train_lrhs))
#         train_hrms = torch.Tensor(np.array(train_hrms))
#         print(train_hrhs.shape, train_hrms.shape)
#         self.train_hrhs_all = train_hrhs
#         self.train_hrms_all = train_hrms
#         self.train_lrhs_all = train_lrhs
#
#     def __getitem__(self, index):
#         train_hrhs = self.train_hrhs_all[index, :, :, :]
#         train_hrms= self.train_hrms_all[index, :, :, :]
#         train_lrhs = self.train_lrhs_all[index, :, :, :]
#         return train_hrhs, train_hrms, train_lrhs
#
#     def __len__(self):
#         return self.train_hrhs_all.shape[0]


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


def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,iteraion, lr_decay_iter, max_iter, power):  ##更新学习率
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr = init_lr1+iteraion/warm_iter*(init_lr2-init_lr1)
    else:
        lr = init_lr2*(1 - (iteraion-warm_iter)/(max_iter-warm_iter))**power
    # param_groups是一个字典，存放优化器的各个参数 {'params': , 'lr': , 'betas': , 'eps': , 'weight_decay': , 'amsgrad': }
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 更新学习率
    return lr


def SAM1(output, target):
    sum1 = output * target
    sum2 = torch.sum(sum1, dim=0)+1e-10  # 对tensor某一维度的数据进行求和
    norm_abs1 = torch.sqrt(torch.sum(output*output,dim=0)) + 1e-10
    norm_abs2 = torch.sqrt(torch.sum(target*target,dim=0)) + 1e-10
    aa = sum2/norm_abs1/norm_abs2
    aa[aa<-1] = -1
    aa[aa>1] = 1
    spectralmap = torch.acos(aa)
    return torch.mean(spectralmap)


def rmse1(Fuse1, HRHSI):
    ap = []
    ae = []
    for j in range(Fuse1.shape[0]):
        be = np.mean(np.square(
            np.float64(np.uint8(np.round(Fuse1[j, ...] * 255))) - np.float64(np.uint8(np.round(HRHSI[j, ...] * 255)))))

        bp = 10 * np.log10((255 ** 2) / be)
        ap.append(bp)
        ae.append(be)

    temp_rmse = np.sqrt(np.mean(np.array(ae)))
    temp_psnr = np.mean(np.array(ap))
    return temp_rmse, temp_psnr

def reconstruction(net2, MSI, HSI, training_size, stride, downsample_factor):
    index_matrix = torch.zeros((31, MSI.shape[2], MSI.shape[3])).cuda()  # [B, C, H, W]
    abundance_t = torch.zeros((31, MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            with torch.no_grad():
                HRHSI = net2(temp_hrms.cuda(), temp_lrhs.cuda())
                HRHSI = HRHSI.squeeze()
                HRHSI = torch.clamp(HRHSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HRHSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]
    HSI_recon = abundance_t / index_matrix
    return HSI_recon