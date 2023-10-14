



import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import copy
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate,test_inference
from models import MLP,CNNMnist,CNNfashion_Mnist,CNNCifar
from utils import get_dataset,exp_details,average_weights


if __name__ == '__main__':
    start_time = time.time()

    # 定义路径
    path_project = os.path.abspath('../..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    #加载数据集和用户组
    train_dataset,test_dataset,user_groups = get_dataset(args)

    # 创建模型
    if args.model == 'cnn':
        # 选择卷积神经网络
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNfashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # 多层感知机
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in,dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('错误，没有此模型')

    #  设置训练模型并且发送给设备
    global_model.to(device)
    global_model.train()
    print(global_model)

    # 复制权重
    global_weights = global_model.state_dict()

    #训练
    train_loss,train_accuracy = [], []
    val_acc_list,net_list = [],[]
    cv_loss,cv_acc = [],[]
    print_every = 2
    val_loss_pre,counter = 0 , 0

    for epoch in tqdm(range(args.epochs)):
        local_weights,local_losses = [],[]
        print(f'\n | 全局训练轮次 : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users) , 1 )
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args,dataset=train_dataset,
                                      idxs=user_groups[idx],logger=logger)
            # w,loss = local_model.update_weights(
            #     model=copy.deepcopy(global_model),global_round=epoch)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # 更新全局权重
        global_weights = average_weights(local_weights)

        # ????
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        #计算每个轮次所有用户的平均训练精度
        list_acc,list_loss = [],[]
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args,dataset=train_dataset,
                                      idxs=user_groups[c],logger=logger)
            acc,loss = local_model.inference(model = global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_loss)/len(list_acc))

        # 每轮次结束输出全局训练损失
        if (epoch+1) % print_every == 0:
            print(f' \n Avg {epoch+1} 轮全局训练数据统计:')
            print(f'训练损失 : {np.mean(np.array(train_loss))}')
            print('训练精度: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # 训练结束, 测试
    test_acc,test_loss = test_inference(args,global_model,test_dataset)

    print(f' \n {args.epochs} 轮全局训练后的结果:')
    print("|---- Avg 训练精度: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- 测试精度: {:.2f}%".format(100*test_acc))

    # 保存目标训练损失和训练精度
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
        args.dataset,args.model,args.epochs,args.frac,args.iid,
            args.local_ep,args.local_bs)

    with open(file_name,'wb') as f:
        pickle.dump([train_loss,train_accuracy],f)

    print('\n 累计运行时间: {0:0.4f}'.format(time.time() - start_time))

    #保存图像曲线信息
    matplotlib.use('Agg')

    # 绘制损失曲线
    plt.figure()
    plt.title('训练损失与通信轮次')
    plt.plot(range(len(train_loss)),train_loss,color='r')
    plt.ylabel('训练损失')
    plt.xlabel('通讯轮次')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(
        args.dataset,args.model,args.epochs,args.frac,args.iid,
        args.local_ep,args.local_bs))

    plt.figure()
    plt.title('Avg 精度与通信轮次')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Avg 精度')
    plt.xlabel('通讯轮次')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs))

