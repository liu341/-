import copy
import torch
from torchvision import datasets,transforms
from sampling import mnist_iid,mnist_noniid,mnist_noniid_unequal
from sampling import cifar_iid,cifar_noniid






def get_dataset(args):
    """ 返回训练和测试数据集以及一个用户组，
    该用户组是一个dict，其中键是用户索引，
    值是每个用户的相应数据。
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar'
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