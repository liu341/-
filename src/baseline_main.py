



import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP,CNNMnist,CNNfashion_Mnist,CNNCifar


if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # 加载数据集
    train_dataset,test_dataset,_ = get_dataset(args)

    # 创建模型
    if args.model == 'cnn':
        # 选择卷积神经网络
        if args.dataset == 'mnist':
            globa_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            globa_model = CNNfashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            globa_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # 多层感知机
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in,dim_hidden=64,dim_out=args.num_classes)

    else:
        exit('错误，没有此模型')

    # 设置训练模型并且发送给设备
    global_model.to(device)
    global_model.train()
    print(global_model)

    # 训练
    # 设置优化器参数
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(),lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adm':
        optimizer = torch.optim.Adam(global_model.parameters(),lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx,(images,labels) in enumerate(trainloader):
            images,labels = images.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            if(batch_idx % 50 == 0):
                print('训练轮次: {}[{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                    epoch+1,batch_idx*len(images),len(trainloader.dataset),
                    100. * batch_idx / len(trainloader),loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\n 训练损失： ',loss_avg)
        epoch_loss.append(loss_avg)

    # 图像展示
    plt.figure()
    plt.plot(range(len(epoch_loss)),epoch_loss)
    plt.xlabel('训练轮次')
    plt.ylabel('训练损失')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(
        args.dataset,args.model,args.epochs))


    # 测试数据集
    test_acc,test_loss = test_inference(args,global_model,test_dataset)
    print('对 ', len(test_dataset), '个样本测试')
    print("测试精确度:{:.2f}%".format(100*test_acc))
    






