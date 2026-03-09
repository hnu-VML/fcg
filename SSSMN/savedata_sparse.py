from Utils import *
from scipy.io import loadmat
from scipy.io import savemat
import time

PSF = fspecial('gaussian', 7, 3)
R = create_F()
downsample_factor = 16
CUDA = 'cuda:1'
torch.cuda.set_device(1)
dataset = 'harvard'  # cave
model_weight_path = './run_new/har_result/92_harvard.pkl'

device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
model = torch.load(model_weight_path, map_location=device).eval()
# print(model)
time_sum=[]
if dataset == 'harvard':
    path2 = '/home/ps/fcg/fcg/code/harvard/test/'
    save_path = './result/harvard/'
else:
    path2 = '/home/ps/fcg/fcg/code/cave/test/'
    save_path = './result/cave/'
imglist = os.listdir(path2)
for i in range(len(imglist)):
    img1 = loadmat(path2 + imglist[i])['b']
    HRHSI = np.transpose(img1, (2, 0, 1))  # [H, W, C] -> [C, H, W]
    MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
    HSI = Gaussian_downsample(HRHSI, PSF, downsample_factor)  # 高斯下采样，使图像模糊,降低分辨率
    MSI_1 = torch.Tensor(np.expand_dims(MSI, axis=0))  # 扩展数组的形状
    HSI_1 = torch.Tensor(np.expand_dims(HSI, axis=0))
    time_s = time.time()
    prediction = reconstruction(model, MSI_1, HSI_1, 128, 64, downsample_factor)  # [C, H, W]
    time_e = time.time()-time_s
    time_sum.append(time_e)

    Fuse = prediction.cpu().detach().numpy()
    Fuse = np.transpose(Fuse, (1, 2, 0))  # [C, H, W] - > [H, W, C]
    savemat(save_path + imglist[i], {'b': Fuse})
    print(i)
print(np.mean(time_sum))
