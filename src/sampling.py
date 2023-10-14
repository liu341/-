import numpy as np
from torchvision import datasets,transforms







def mnist_iid(dataset,num_users):
    """
    为用户生成 IID 数据 数据集为mnist
    :param dataset:  数据集
    :param num_users: 用户数量
    :return:
    """
    num_items= int(len(dataset)/num_users)
    dict_users,all_idxs = {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def mnist_noniid(dataset, num_users):
    """
    为用户生成 NON-IID 数据 数据集为mnist
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 训练图片，划分为200 个大小为300 的分片
    num_shards,num_imgs = 200,300
    idx_shard = [i for i in range(num_shards)] # 生成了一个 [0,1,2,....200]
    dict_users = {i: np.array([]) for i in range(num_users)} # 生成了 200 个array
    idxs = np.arange(num_shards*num_imgs) # 生成了一个[0 1 2 3 4 ... 60,000]
    labels = dataset.train_labels.numpy() # 将标签 作为 []

    # 标签排序
    idxs_labels = np.vstack((idxs , labels)) #将两个数组合并为一个
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] # 按标签从小到大排序，同时idxs随着标签的排序一并排序
    idxs = idxs_labels[0,:] # 可以理解为 标签的原下标

    # 划分数据集，并为每个客户端分配两个分片
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard,2,replace=False)) # 随机选两个数字 ， 即为分片的下表
        idx_shard = list(set(idx_shard) - rand_set) # 移除这两个数字
        for rand in rand_set: # 便利随机选的两个数字
            dict_users[i] = np.concatenate(
                (dict_users[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),axis=0) # 选择对应的分片 concatenate 连接作用
    return  dict_users


def mnist_noniid_unequal(dataset,num_users):
    """
    非平均数据量的数据集分配  nonIID数据，Mnist 数据集
    60，000 张训练图像，1200 个分片，每个分片50个数据样本
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards,num_imgs = 1200,50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i : np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # 标签排序
    idxs_labels = np.vstack((idxs,labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # 定义最小分片数和最大分片数
    min_shard = 1
    max_shard = 30

    #随机为每一个客户端分块
    #特别注意：所有块的总和为 num_shards
    random_shard_size = np.random.randint(min_shard,max_shard+1,size=num_users)

    random_shard_size = np.around(random_shard_size/sum(random_shard_size)*num_shards)

    random_shard_size = random_shard_size.astype(int)

    # 随机为每个客户端分配分片
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # 首先为每一个客户端分配一个分片，
            # 保证每个客户端都有不少于一的分片个数
            rand_set = set(np.random.choice(idx_shard,1,replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size -= 1

        #随后随机分配剩下的分片
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            shard_size = min(shard_size,len(idx_shard))

            rand_set = set(np.random.choice(idx_shard,shard_size,replace=False))

            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in  range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard,shard_size,replace=False))

            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0 :
            shard_size = len(idx_shard)
            #  将剩余的碎片添加到数据最少的客户端
            k = min(dict_users,key=lambda x:len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard,shard_size,replace=False))

            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[k],idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users





def cifar_iid(dataset,num_users):
    """

    :param dataset:
    :param num_users:
    :return:
    """
    num_items = int(len(dataset)/num_users)
    dict_users,all_idxs = {},[i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))

        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def cifar_noniid(dataset,num_users):
    """

    :param dataset:
    :param num_users:
    :return:
    """
    num_shards,num_imgs = 200,250
    idx_shard = [i for i in range(num_shards)]
    dict_user = {i : np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    #
    labels = np.array(dataset.targets)

    # 标签排序
    idxs_labels = np.vstack((idxs,labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    #分配
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard,2,replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_user[i] = np.concatenate(
                (dict_user[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),axis=0)
    return dict_user


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/',train=True,download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    labels = dataset_train.train_labels.numpy()
    for i in range(10):
        print(labels[i])
    num = 100
    d = mnist_noniid(dataset_train,num)