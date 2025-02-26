from Utils import *
from Net import SSTF_Unet  # 对MSI应用空间自注意力，对HSI应用光谱自注意力
import torch.nn as nn
import math

if __name__ == '__main__':
    PSF = fspecial('gaussian', 7, 3)
    R = create_F()  # 光谱波段采样矩阵
    training_size = 128
    downsample_factor = 16  # 空间下采样倍数
    EPOCH = 100  # 一个epoch为全部样本训练一次
    BATCH_SIZE = 32  # 每一次训练的样本数
    decay_power = 1.5  # 衰减指数
    learning = 1  # 选择学习率更新方案 0:poly 1:warm
    dataset = 'cave'  # cave

    if dataset == 'harvard':
        CUDA = 'cuda:0'
        torch.cuda.set_device(0)
        stride = 40  # 步长 harvard: 40
        LR = 5e-4  # 指定学习率
        init_lr1 = 1e-4
        init_lr2 = 5e-4
        high = 1392
        width = 1040
        model_name = 'harvard'
        path1 = './harvard/train/'
        path2 = './harvard/test/'
        path3 = './run/har_result/'
        num = 35  # 训练集的HSHRI数量
    else:
        CUDA = 'cuda:1'
        torch.cuda.set_device(1)
        stride = 16  # 步长 cave: 16
        LR = 1e-3  # 指定学习率
        init_lr1 = 1e-4
        init_lr2 = 1e-3
        high = 512
        width = 512
        model_name = 'cave'
        path1 = './cave/train/'
        path2 = './cave/test/'
        path3 = './run/cave_result/'
        num = 24  # 训练集的HSHRI数量

    imglist_test = os.listdir(path2)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表

    device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

    maxiteration = math.ceil(((high - training_size) // stride + 1) * ((width - training_size) // stride + 1) * num / BATCH_SIZE) * EPOCH  ##向上取整
    warm_iter = math.floor(maxiteration / 30)  ##向下取整

    train_data = HSI_MSI_Data(path1)  # 实例化HSI_MSI_Data类
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)  # 数据加载器

    model = SSTF_Unet(in_channels_MSI=3, out_channels=31, n_feat=31).to(device)

    # 选择损失函数与优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    l1_loss = nn.L1Loss(reduction='mean')  # 取预测值和真实值的绝对误差的平均数

    step = 0
    # loss_list = []
    for epoch in range(EPOCH):  # 开始训练
        for step1, (a1, a2, a3) in enumerate(train_loader):  # hrhs, hrms, lrhs (type:tensor)
            model.train()  # 训练模式
            if learning == 0:
                lr = poly_lr_scheduler(optimizer, LR, step, max_iter=maxiteration, power=decay_power)  # update lr
            else:
                lr = warm_lr_scheduler(optimizer, init_lr1, init_lr2, warm_iter, step, lr_decay_iter=1,  max_iter=maxiteration, power=decay_power)  # 得到学习率
            step = step + 1
            output = model(a2.cuda(), a3.cuda())
            result = torch.clamp(output, 0, 1)
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

            # LOSS.append(np.mean(np.abs(HRHSI-Fuse)))


        torch.save(model, path3 + str(epoch) + '_' + model_name + '.pkl')  # 每训练一个epoch,保存一次model
        print(f'epoch：{epoch}   lr：{lr}    RMSE：{np.mean(RMSE)}    SAM：{np.mean(SAM)}')
#        torch.save(model.state_dict(), dataset + '.pkl')  # 只保存模型的参数
