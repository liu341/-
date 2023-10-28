import copy
import os.path

import numpy as np
import torch
from torchvision import datasets,transforms
from sampling import mnist_iid,mnist_noniid,mnist_noniid_unequal
from sampling import cifar_iid,cifar_noniid


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dataset(args):
    """ 返回训练和测试数据集以及一个用户组，
    该用户组是一个dict，其中键是用户索引，
    值是每个用户的相应数据。
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar'
        if args.paper =='FedHQ':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32,padding=4),#在一个随机的位置进行裁剪
                transforms.RandomHorizontalFlip(),#以0.5的概率水平翻转给定的PIL图像
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))#将一个tensor image根据其均值和方差进行归一化
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=transform_train)

            test_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                            transform=transform_test)
        else:
            apply_transfrom = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

            train_dataset = datasets.CIFAR10(data_dir,train=True,download=True,
                                             transform=apply_transfrom)

            test_dataset = datasets.CIFAR10(data_dir,train=True,download=True,
                                            transform=apply_transfrom)

        # 用户之间的样本培训数据
        if args.iid:
            # IID 数据
            user_groups = cifar_iid(train_dataset,args.num_users)
        else:
            # 非IID 数据
            if args.unequal:
                # 选择不平均切割
                raise NotImplementedError()
            else:
                # 选择对等切割为每一个用户
                user_groups = cifar_noniid(train_dataset,args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))])

        train_dataset = datasets.MNIST(data_dir,train=True,download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir,train=True,download=True,
                                      transform=apply_transform)

        # 用户之间的样本训练数据
        if args.iid:
            # IID 数据
            user_groups = mnist_iid(train_dataset,args.num_users)
        else:
            # 非IID 数据
            if args.unequal:
                user_groups = mnist_noniid_unequal(train_dataset,args.num_users)
            else:
                user_groups = mnist_noniid(train_dataset,args.num_users)



    return train_dataset,test_dataset,user_groups

def get_quantization_bit(args):
    quant_bit_for_user = np.zeros([args.num_users])
    avg_for_user = np.zeros([args.num_users])

    for user in range(args.num_users):
        if user < args.num_users * args.bit_4_ratio:
            quant_bit_for_user[user] = 4
            avg_for_user[user] = 4
        elif user < args.num_users * (args.bit_4_ratio + args.bit_8_ratio):
            quant_bit_for_user[user] = 8
            avg_for_user[user] = 8
        else:
            avg_for_user[user] = 64

    return quant_bit_for_user,avg_for_user


def average_weights(w):
    """
        返回平均权重
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key],len(w))
    return w_avg

def average_weights_HQ(args,w,avg_for_user,q_for_user):
    w_avg= copy.deepcopy(w[0])
    user_num = len(avg_for_user)
    # 计算p值为每一个用户，并且P值和为1
    p_for_user = np.ones(user_num)
    if args.average_scheme == 'FedAvg':
        p_for_user /= sum(p_for_user)
    if args.average_scheme == 'Proportional':
        p_for_user = np.array(avg_for_user) / np.sum(avg_for_user)
    if args.average_scheme == 'FedHQ':
        p_for_user = 1/(1+q_for_user)
        p_for_user = np.sum(p_for_user)
    for key in w_avg:
        w_avg[key] = w_avg[key].float()
        w_avg[key] *= p_for_user[0]
    for i in range(1,len(w)):
        weight_name = []
        for j in w[i]:
            weight_name.append(j)
        cnt = 0
        for key in w_avg.keys():
            w_avg[key] += w[i][weight_name[cnt]] * p_for_user[i]
            cnt += 1
    return w_avg

def exp_details(args):
    print('\n实验细节:')
    print(f'   模型:      {args.model}')
    print(f'  优化方案:    {args.optimizer}')
    print(f'   学习率:     {args.lr}')
    print(f' 全局训练轮次:  {args.epochs}\n')

    print('  联邦参数:  ')
    if args.iid:
        print('    IID    ')
    else:
        print('  Non-IID  ')
    print(f'   用户比例:   {args.frac}')
    print(f'   本次批次:   {args.local_bs}')
    print(f' 本地训练轮次:  {args.local_ep}\n')
    return 