import argparse
# 参数配置表
def args_parser():
    parser = argparse.ArgumentParser()

    # 联邦学习参数
    parser.add_argument('--epochs',type=int,default=1,help="训练轮次")
    parser.add_argument('--num_users',type=int,default=100,help="用户数量 K")
    parser.add_argument('--frac',type=float,default=0.1,help='客户端比例 C')
    parser.add_argument('--local_ep',type=int,default=1,help='局部训练次数 E')
    parser.add_argument('--local_bs',type=int,default=50,help='本地批量大小 B')
    parser.add_argument('--lr',type=float,default=0.01,help='学习率')
    parser.add_argument('--momentum',type=float,default=0.5,help='SGD 动量')
    # FedHq
    parser.add_argument('--weight_decay',type=float,default=0.0005,help='权重衰减')
    parser.add_argument('--average_scheme',type=str,default='FedHQ',help='选择平均方案',choices=['FedAvg','Proportional','FedHQ'])
    parser.add_argument('--quant_bits',type=int,default=8,help='记录当前量化位')
    parser.add_argument('--bit_4_ratio',type=float,default=0.6,help='4位客户端的比例')
    parser.add_argument('--bit_8_ratio',type=float,default=0.4,help='6位客户端的比例')
    parser.add_argument('--quant_type',type=str,default='stochastic',metavar='S',help='四舍五入法，随机或接近',choices=['stochastic','nearest'])


    # 模型参数
    parser.add_argument('--model',type=str,default='mlp',help='模型名称')
    parser.add_argument('--kernel_num',type=int,default=9,help='内核数量')
    parser.add_argument('--kernel_sizes',type=str,default='3,4,5',help='卷积大小')
    parser.add_argument('--num_channels',type=int,default=1,help='图像通道数')
    parser.add_argument('--norm',type=str,default='batch_norm',help='batch_norm, layer_norm, or None 批规范，层规范或者无')
    parser.add_argument('--num_filters',type=int,default=32,help='卷积网络的数量，32 for mini-imagent,64 for omiglot')
    parser.add_argument('--max_pool',type=str,default='True',help='是否使用最大池化层而不使用跨步卷积')

    #其他参数
    parser.add_argument('--dataset',type=str,default='mnist',help='数据集名称')
    parser.add_argument('--num_classes',type=int,default=10,help='类别数，预测结果')
    parser.add_argument('--gpu',default=None,help='使用GPU的Id，默认使用CPU')
    parser.add_argument('--optimizer',type=str,default='sgd',help='优化器类型')
    parser.add_argument('--iid',type=int,default=1,help='默认是独立同分布，设置为0可置为非独立同分布')
    parser.add_argument('--unequal',type=int,default=0,help='是否对非i.i.d设置使用不相等的数据拆分（对相等的拆分使用0）')
    parser.add_argument('--stopping_round',type=int,default=10,help='提前停止的轮次')
    parser.add_argument('--verbose',type=int,default=1,help='日志详细信息')
    parser.add_argument('--seed',type=int,default=1,help='随机种子')

    parser.add_argument('--paper',type=str,default='other',choices=['other','FedHQ'])
    args = parser.parse_args()
    return args
