import copy
import os
import time

import numpy as np
import torch
from tqdm import tqdm
import quantizer as qn
import prettytable as pt

from tensorboardX import SummaryWriter

from src.update import LocalUpdate_HQ,test_inference
from src.models import CNNCifar,CNNMnist,CNNfashion_Mnist,CNNCifar_HQ,ResNet,CNNMnist_HQ
from src.options import args_parser
from src.utils import exp_details, get_dataset,get_quantization_bit,make_dir,average_weights_HQ

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    start_time = time.time()

    # 定义路径
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args) # utils 中的方法， 打印配置信息

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # 加载数据集
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.dataset == 'mnist':
        quant = lambda: qn.BlockQuantizer(args.quant_bits,args.quant_bits,args.quant_type)
        global_model = CNNCifar_HQ(args=args,quant=quant)
    elif args.dataset == 'cifar':
        quant = lambda: qn.BlockQuantizer(args.quant_bits,args.quant_bits,args.quant_type)
        quantx = lambda x: qn.BlockQuantizer(x,args.quant_bits,args.quant_bits,args.quant_type)
        if args.iid == 1:
            global_model = ResNet(args=args,quant=quant,quantx=quantx)
        else:
            global_model = CNNCifar_HQ(args=args,quant=quant)

    # 从服务器获取全局权重
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    # 记录损失和精确度
    train_loss,train_accuracy = [],[]
    print_every = 1
    acc_level = np.array(list(range(41)))*0.01+0.6
    acc_table_line = []
    acc_true = []
    acc_flag = np.zeros_like(acc_level)
    quant_bit_for_user ,avg_for_user = get_quantization_bit(args)
    last_max_acc = 0
    # 配置文件名字以保存文件
    result_base_filename = 'result/'+args.dataser+ '/iid/' + args.average_scheme
    if args.iid == 0:
        result_base_filename = 'result/' + args.dataser + '/noniid/' + args.average_scheme
    save_acc_filename = result_base_filename + '/c' + str(int(args.frac*10)) + 'result_04-' + str(int(args.bit_4_ratio*10)) + '_8-' + str(int(args.bit_8_ratio*10)) + '.txt'
    save_pkl_filename = result_base_filename + '/c' + str(int(args.frac*10)) + 'result_04-' + str(int(args.bit_4_ratio*10)) + '_8-' + str(int(args.bit_8_ratio*10)) + '.pkl'

    make_dir(result_base_filename)
    test_accuracy = []
    for epoch in tqdm(range(args.epochs)):
        local_weights,local_losses = [],[]
        print(f'\n | 全局训练轮次: {epoch + 1} | \n')
        m = max(int(args.frac * args.num_users),1)
        idxs_user = np.random.choice(range(args.num_users),m,replace=False)
        cur_q = []
        for idx in idxs_user:
            quantx = None
            quantx = None
            idx = int(idx)
            #为每个客户端设置量化条件和局部模型
            if quant_bit_for_user[idx] != 0:
                args.quant_bits = quant_bit_for_user[idx]
                quant = lambda : qn.BlockQuantizer(args.quant_bits,args.quant_bits,args.quant_type)
                quantx = lambda x:qn.quantize_block(x,args.quant_bits,args.quant_bits,args.quant_type)
            if args.dataset == 'mnist':
                user_model = CNNMnist_HQ(args=args,quant=quant)
            elif args.dataset == 'cifar' and args.iid == 1:
                user_model = ResNet(args=args,quant=quant,quantx=quantx)
            elif args.dataset == 'cifar' and args.iid == 0:
                user_model = CNNCifar_HQ(args=args,quant=quant)
            user_model.to(device)
            user_model.train()
            # 更新局部到全局的权重
            weight_name = []
            for i in global_weights:
                weight_name.append(i)
            cnt = 0
            user_weights = user_model.state_dict()
            for i in user_weights:
                if quantx != None:
                    user_weights[i] = quantx(global_weights[weight_name[cnt]].to(float))
                else:
                    user_weights[i] = global_weights[weight_name[cnt]]
                cnt += 1
            user_model.load_state_dict(user_weights)
            # 训练局部模型
            local_model = LocalUpdate_HQ(args=args,dataset=train_dataset,
                                         idxs=user_groups[idx],logger=logger,quant=quantx,quantbit=args.quant_bits,mode=args.quant_type)
            w,loss,q = local_model.update_weights(model=copy.deepcopy(user_model))
            cur_q.append(q)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            torch.cuda.empty_cache()


        for i in range(len(local_weights)):
            for j in range(1):
                name = "features." + str(j) + ".weight"
                local_weights[i][name] = local_weights[i][name] + torch.randn(local_weights[i][name].shape).to(device)

        global_weights = average_weights_HQ(args,local_weights,avg_for_user[idxs_user],q_for_user=np.array(cur_q))
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        #计算每个轮次所有客户端的平均训练精度
        list_acc,list_loss = [],[]
        weight_name = []
        for i in global_weights:
            weight_name.append(i)
        for c in range(args.num_users):
            quant = None
            quantx = None
            #为每个客户端设置量化条件和本地模型
            if quant_bit_for_user[c] != 0:
                args.quant_bits = quant_bit_for_user[c]
                quant = lambda: qn.BlockQuantizer(args.quant_bits,args.quant_bits,args.quant_type)
                quantx = lambda x: qn.quantize_block(x,args.quant_bits,args.quant_bits,args.quant_type)
            if args.dataset == 'mnist':
                user_model = CNNMnist_HQ(args=args,quant=quant)
            elif args.dataset == 'cifar' and args.iid == 1:
                user_model = ResNet(args=args,quant=quant,quantx=quantx)
            elif args.dataset == 'cifar' and args.iid == 0:
                user_model = CNNCifar_HQ(args=args,quant=quant)
            user_model.to(device)
            user_model.eval()
            cnt = 0
            user_weights = user_model.state_dict()
            for i in user_weights:
                user_weights[i] = global_weights[weight_name[cnt]]
                cnt += 1
            user_model.load_state_dict(user_weights)
            local_model = LocalUpdate_HQ(args=args,dataset=train_dataset,
                                         idxs=user_groups[c],logger=logger,quant=quantx,quantbit=quant_bit_for_user[c],mode=args.quant_type)
            acc,loss = local_model.inference(model=user_model)
            list_acc.append(acc)
            list_loss.append(loss)
            torch.cuda.empty_cache()
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if args.dataset == 'mnist':
            global_model = CNNMnist_HQ(args=args,quant=None)
        elif args.dataset == 'cifar' and args.iid == 1:
            global_model = ResNet(args=args,quant=quant,quantx=quantx)
        elif args.dataset == 'cifar' and args.iid == 0:
            global_model = CNNCifar_HQ(args=args,quant=quant)
        cnt = 0
        current_weight = global_model.state_dict()
        for i in current_weight:
            current_weight[i] = global_weights[weight_name[cnt]]
            cnt += 1
        global_model.load_state_dict(current_weight)
        global_model.to(device)
        global_model.eval()
        test_acc,test_loss = test_inference(args,global_model,test_dataset)
        test_accuracy.append(test_acc)
        # 以最大精度保存到服务器
        if test_acc > last_max_acc:
            last_max_acc = test_acc
            torch.save(global_model.state_dict(),save_pkl_filename)
        with open(save_acc_filename,'a') as file:
            write_content = str(epoch+1) + ' ' + str(np.mean(np.array(train_loss))) + ' ' + str(train_accuracy[-1]) + ' ' + str(test_acc) + '\n'
            file.write(write_content)
        for acc_index in range(acc_level):
            if acc_flag[acc_index] == 0 and test_acc >= acc_level[acc_index] and test_acc not in acc_true and test_acc >= acc_level[acc_index+1]:
                acc_flag[acc_index] = 1
            if acc_flag[acc_index] == 0 and test_acc >= acc_level[acc_index] and test_acc not in acc_true:
                acc_true.append(test_acc)
                acc_flag[acc_index] = 1
                acc_table_line.append(epoch+1)
        if (epoch+1) % print_every == 0:
            print(f' \n {epoch+1}轮全局训练数据:')
            print(f'训练损失:{np.mean(np.array(train_loss))}')
            print('训练精度:{:.2f}%'.format(100*train_accuracy[-1]))
            print('测试精度:{:.2f}%'.format(100*test_acc))
            print(f'测试损失:{np.mean(np.array(test_loss))}')
            print('4: ',args.bit_4_ratio,', 8: ',args.bit_8_ratio,', ',args.average_scheme)
            if len(acc_table_line) == 0:
                print('错误！无法获得目标精度的通信回合')
            else:
                table = pt.PrettyTable()
                table.field_names = acc_true
                table.add_row(acc_table_line)
                print(table)
        if (epoch+1) % 10 == 0 and args.lr >= 1e-4:
            args.lr = args.lr * 0.9
        print('学习率:',args.lr)
    # 测试
    test_acc,test_loss = test_inference(args,global_model,test_dataset)
    print(f' \n {args.epochs}轮训练后的结果:')
    print("|---- 平均训练精度: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- 测试精度: {:.2f}%".format(100*test_acc))
    print('\n 总计运行时间:{0:0.4f}'.format(time.time() - start_time))
    table = pt.PrettyTable()
    table.field_names = acc_true
    table.add_row(acc_table_line)
    test_acc = 0
    if len(acc_table_line) == 0:
        print('错误！无法获得目标精度的通信回合')
    else:
        table = pt.PrettyTable()
        table.field_names = acc_true
        table.add_row(acc_table_line)
        print(table)
    #return train_accuracy[-1],test_acc











