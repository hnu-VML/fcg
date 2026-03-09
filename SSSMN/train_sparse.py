from Utils import *
from models.vmunet.vmamba_sparse import VSSM
import torch.nn as nn
import math
import os 
import pandas as pd
import time
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    seed = 42
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    PSF = fspecial('gaussian', 7, 3)
    R = create_F()  # 光谱波段采样矩阵
    training_size = 128
    downsample_factor = 16  # 空间下采样倍数
    EPOCH = 100  # 一个epoch为全部样本训练一次
    BATCH_SIZE = 8  # 每一次训练的样本数
    decay_power = 1.5  # 衰减指数
    learning = 1  # 选择学习率更新方案 0:poly 1:warm
    LR = 2e-4  # 指定学习率1e-3
    init_lr1 = 2e-5
    init_lr2 = 2e-4
    dataset = 'cave'  # cave
    continue_learn=False
    if continue_learn:
        start_epoch=41  # start train
        model_weight_path = './run_new/cave_result/40_cave.pkl' # start_epoch前保存下的权重 
        model = torch.load(model_weight_path).cuda()
    else:
        start_epoch=0
        model = VSSM(patch_size=2, num_classes=31, d_state_spe=8, depths=[1,1,1], depths_decoder=[1,1,1], c_dim=34, dims=[68, 136, 272], dims_decoder=[272, 136, 68], drop_path_rate=0.1).cuda()

    if dataset == 'harvard':
        stride = 40  # 步长 harvard: 40
        high = 1392
        width = 1040
        model_name = 'harvard'
        path1 = '/home/ps/fcg/fcg/code/harvard/train/'
        path2 = '/home/ps/fcg/fcg/code/harvard/test/'
        path3 = './run_new/har_result/'
        num = 35  # 训练集的HSHRI数量
    else:
        stride = 16  # 步长 cave: 16
        high = 512
        width = 512
        model_name = 'cave'
        path1 = '/home/ps/fcg/fcg/code/cave/train/'
        path2 = '/home/ps/fcg/fcg/code/cave/test/'
        path3 = './run_new/cave_result/'
        num = 24  # 训练集的HSHRI数量
    
    # 创建excel
    excel_name = model_name+'_record_sparse.csv'
    excel_path = os.path.join('./run_new', excel_name)
    if not os.path.exists(excel_path):
        df = pd.DataFrame(columns=['epoch', 'lr','val_psnr', 'val_rmse', 'val_sam'])  # 列名
        df.to_csv(excel_path, index=False)

    imglist_test = os.listdir(path2)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表

    # device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

    maxiteration = math.ceil(((high - training_size) // stride + 1) * ((width - training_size) // stride + 1) * num / BATCH_SIZE) * EPOCH  ##向上取整
    warm_iter = math.floor(maxiteration / 30)  ##向下取整

    train_data = HSI_MSI_Data(path1)  # 实例化HSI_MSI_Data类
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)  # 数据加载器

    # 选择损失函数与优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    l1_loss = nn.L1Loss(reduction='mean')   # 取预测值和真实值的绝对误差的平均数
    # fft_loss = FFTLoss(loss_weight = 0.05, reduction='mean')

    step=(len(os.listdir(path1))//BATCH_SIZE)*start_epoch
    # loss_list = []
    for epoch in range(start_epoch, EPOCH):  # 开始训练
        for step1, (a1, a2, a3) in enumerate(train_loader):  # hrhs, hrms, lrhs (type:tensor)
            model.train()  # 训练模式
            if learning == 0:
                lr = poly_lr_scheduler(optimizer, LR, step, max_iter=maxiteration, power=decay_power)  # update lr
            else:
                lr = warm_lr_scheduler(optimizer, init_lr1, init_lr2, warm_iter, step, lr_decay_iter=1,  max_iter=maxiteration, power=decay_power)  # 得到学习率
            step = step + 1
            output = model(a2.cuda(), a3.cuda())
            result = torch.clamp(output, 0, 1)
            #loss = l1_loss(result, a1.cuda()) + fft_loss(result, a1.cuda())
            loss = l1_loss(result, a1.cuda())
            # loss_list.append(loss.item())

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新权重
        # print('Epoch'+str(epoch)+': '+'LOSS='+str(np.mean(loss_list)))
        # loss_list.clear()
        model.eval()  # 测试模型
        RMSE = []
        SAM = []
        PSNR=[]
        # LOSS = []
        for i in range(len(imglist_test)):
            img1 = loadmat(path2 + imglist_test[i])['b']
            HRHSI = np.transpose(img1, (2, 0, 1))  # [H, W, C] -> [C, H, W]
            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            HSI = Gaussian_downsample(HRHSI, PSF, downsample_factor)  # 高斯下采样，使图像模糊,降低分辨率
            MSI_1 = torch.Tensor(np.expand_dims(MSI, axis=0))  # 扩展数组的形状
            HSI_1 = torch.Tensor(np.expand_dims(HSI, axis=0))
            HRHSI_1 = torch.Tensor(HRHSI)

            prediction = reconstruction(model, MSI_1, HSI_1, training_size, 64, downsample_factor)  # [C, H, W]

            sam = SAM1(prediction, HRHSI_1.cuda())
            sam = sam * 180 / math.pi
            SAM.append(sam.cpu().detach().numpy())

            Fuse = prediction.cpu().detach().numpy()
            a, b = rmse1(Fuse, HRHSI)
            RMSE.append(a)
            PSNR.append(b)

            # LOSS.append(np.mean(np.abs(HRHSI-Fuse)))


        torch.save(model, path3 + str(epoch) + '_' + model_name + '.pkl')  # 每训练一个epoch,保存一次model
        print(f'epoch：{epoch}   lr：{lr}    PSNR: {np.mean(PSNR)}    RMSE：{np.mean(RMSE)}    SAM：{np.mean(SAM)}')

        val_list = [epoch, lr, np.mean(PSNR), np.mean(RMSE), np.mean(SAM)]
        val_data = pd.DataFrame([val_list])
        val_data.to_csv(excel_path, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
        time.sleep(0.1)
