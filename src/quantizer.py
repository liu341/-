from __future__ import print_function
import torch
from options import args_parser
import copy
import numpy as np
import math
import random

args = args_parser()

def get_bound_avg(data,n_clusters):
    bound = np.zeros(n_clusters + 1)
    # # if epoch == 0:
    min = torch.min(data)
    max = torch.max(data)
    bound[0] = min
    bound[n_clusters] = max
    for i in range(1, n_clusters):
        bound[i] = min + i * (max - min) / n_clusters
    return bound

def get_bound_msqe(n_clusters,weight,bound):
    if n_clusters < weight.shape[0] < 1000:
        return  bound
    bound_msqe = copy.deepcopy(np.array(bound))
    weight_temp = weight.reshape([1, -1])
    X_Sort = np.array(sorted(weight_temp[0]))
    # print(X_Sort)

    for item in range(1000000):
        bound_temp = copy.deepcopy(bound_msqe)
        for i in range(1, n_clusters):
            pa = np.where((X_Sort >= bound_msqe[i - 1]) & (X_Sort <= bound_msqe[i + 1]))
            if len(pa[0]) == 0:
                bound_msqe[i] = bound[i]
            else:
                sum_bqplus = sum(X_Sort[pa[0]])
                temp = (bound_msqe[i + 1] * len(pa[0]) - sum_bqplus) / (bound_msqe[i + 1] - bound_msqe[i - 1])
                # print(temp)
                temp = math.floor(temp)

                bound_msqe[i] = X_Sort[temp + pa[0][0]]
                if bound_msqe[i] >= bound_msqe[i + 1]:
                    bound_msqe[i] = bound[i]
        if sum(bound_msqe == bound_temp) == (n_clusters + 1):
            break

    return bound_msqe

def weight_to_bound(n_clusters,weight,bound):# 把权重量化到边界上
    for k in range(1,n_clusters+1):
        pa = np.where((weight >= bound[k - 1]) & (weight < bound[k]))
        tmp = weight[pa[0]].reshape([-1, 1])
        k_max = bound[k]
        k_min = bound[k - 1]
        if k_min != k_max:
            p_min = (k_max - tmp) / (k_max - k_min)
        else:
            p_min = 1
        prob = np.random.rand(tmp.shape[0], tmp.shape[1])  # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
        tmp[prob < p_min] = k_min  # 随机量化
        tmp[prob >= p_min] = k_max
        weight[pa[0]] = tmp
    return weight

def quantize_msqe_liu(data, bits, q_cal, write, key, epoch):
    n_clusters = pow(2, bits) - 1  # 7
    data_shape = data.shape
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    original_data = copy.deepcopy(data) # 记录原始权重
    bound = get_bound_avg(data,n_clusters) # 平均边界
    bound_msqe = get_bound_msqe(n_clusters,weight,bound) # 量化之后的边界
    weight = weight_to_bound(n_clusters, weight, bound_msqe) # 量化后的权重
    if write == 1: # 记录
        np.set_printoptions(threshold=100000000000)
        f = open('One_note/weight.txt', 'a')
        f.write(str(epoch + 1) + '\n' + 'bound ' + 'msqe' + '\n' + str(bound_msqe.tolist()) + '\n' + str(key) + '\n' +
                str(weight.reshape([1, -1])) + '\n')
        f.write('msqe ' + str(epoch + 1) + ' quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
        f.close()

    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    q = 0
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    else:
        q = q / torch.pow(original_data)

    return quant_data, q




def quantize_msqe(data, bits, q_cal, write, key, epoch):  # data, 3, SQE
    n_clusters = pow(2, bits) - 1  # 7
    data_shape = data.shape
    # np.set_printoptions(suppress=True)
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    original_data = copy.deepcopy(data)
    q = 0
    # weight = weight_normal(weight)
    if n_clusters < weight.shape[0] < 1000:
        bound = np.zeros(n_clusters + 1)
        # # if epoch == 0:
        min = torch.min(data)
        max = torch.max(data)
        bound[0] = min
        bound[n_clusters] = max
        for i in range(1, n_clusters):
            bound[i] = min + i * (max - min) / n_clusters

        bound_msqe = np.array(bound)
        for k in range(1, n_clusters + 1): # 把权重量化到边界上
            pa = np.where((weight >= bound_msqe[k - 1]) & (weight < bound_msqe[k]))
            tmp = weight[pa[0]].reshape([-1, 1])
            k_max = bound_msqe[k]
            k_min = bound_msqe[k - 1]
            if k_min != k_max:
                p_min = (k_max - tmp) / (k_max - k_min)
            else:
                p_min = 1
            prob = np.random.rand(tmp.shape[0], tmp.shape[1])  # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
            tmp[prob < p_min] = k_min  # 随机量化
            tmp[prob >= p_min] = k_max
            weight[pa[0]] = tmp
            original_bound = bound
    if weight.shape[0] >= 1000:  # weight.shape => (row,cal)
        bound = np.zeros(n_clusters + 1)
        min = torch.min(data)
        max = torch.max(data)
        bound[0] = min
        bound[n_clusters] = max
        for i in range(1, n_clusters):
            bound[i] = min + i * (max - min) / n_clusters
        original_bound = bound
        # bound = []
        # estimator = KMeans(n_clusters=n_clusters)
        # estimator.fit(weight)  # 聚类
        # label_pred = estimator.labels_  # 聚类标签
        # all_min = np.zeros(n_clusters)
        # # 该for循环是将各类中最小的数添加到all_min
        # for k in range(n_clusters):
        #     if weight[label_pred == k].shape[0] == 0:  # weight[label_pred == k].shape[0] => 标签k类型的个数
        #         continue
        #     all_min[k] = np.min(weight[label_pred == k])  # all_min[k] 标签k中最小的数值
        # # index是各类最小值的排序
        # index = np.argsort(all_min)
        # cnt = 0
        # last_max = 0
        # # 该for循环是将各类的最小值，和最小值最大的那一类的最大值添加到bound
        # for k in index:  # index 0-6
        #     tmp = weight[label_pred == k]
        #     if tmp.shape[0] == 0:
        #         continue
        #     bound.append(np.min(tmp))
        #     last_max = np.max(tmp)
        #     cnt += 1
        # bound.append(last_max)

        bound_msqe = copy.deepcopy(np.array(bound))
        weight_temp = weight.reshape([1, -1])
        np.set_printoptions(suppress=True)
        input_vector = sorted(weight_temp[0])
        input_vector = np.array(input_vector)
        for iter in range(1000000):
            bound_temp = copy.deepcopy(bound_msqe)
            for i in range(1, n_clusters):
                # temp1 = input_vector - bound_msqe[i-1]
                # temp2 = input_vector - bound_msqe[i+1]
                # pa = np.where((temp1 * temp2) <= 0)
                pa = np.where((input_vector >= bound_msqe[i - 1]) & (input_vector <= bound_msqe[i + 1]))
                if len(pa[0]) == 0:
                    bound_msqe[i] = bound[i]
                else:
                    sum_bqplus = sum(input_vector[pa[0]])
                    temp = (bound_msqe[i + 1] * len(pa[0]) - sum_bqplus) / (bound_msqe[i + 1] - bound_msqe[i - 1])
                    temp = math.floor(temp)

                    bound_msqe[i] = input_vector[temp + pa[0][0]]
                    if bound_msqe[i] >= bound_msqe[i + 1]:
                        bound_msqe[i] = bound[i]
            if sum(bound_msqe == bound_temp) == (n_clusters + 1):
            # if sum(bound_msqe - bound_temp) <= 1e-5:
                break

        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write(str(epoch + 1) + '\n' + 'bound ' + 'msqe' + '\n' + str(bound_msqe.tolist()) + '\n' + str(key) + '\n' +
                    str(weight.reshape([1, -1])) + '\n')
            f.close()
        # 该for循环是先计算各类的p_min,然后根据p_min和随机生成的样本来随机量化data
        cnt = 0
        for k in range(1, n_clusters + 1):
            # temp1 = weight - bound_msqe[k - 1]
            # temp2 = weight - bound_msqe[k]
            # pa = np.where((temp1 * temp2) <= 0)
            pa = np.where((weight >= bound_msqe[k - 1]) & (weight <= bound_msqe[k]))
            tmp = weight[pa[0]].reshape([-1, 1])
            k_max = bound_msqe[k]
            k_min = bound_msqe[k - 1]
            cnt += 1
            if k_min != k_max:
                p_min = (k_max - tmp) / (k_max - k_min)
            else:
                p_min = 1
            prob = np.random.rand(tmp.shape[0], tmp.shape[1])  # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
            tmp[prob < p_min] = k_min  # 随机量化
            tmp[prob >= p_min] = k_max
            weight[pa[0]] = tmp
            # if q_cal == 'expectation':
            #     q = torch.max(q, torch.max(torch.tensor(np.power(original_tmp - tmp, 2) / original_tmp)))
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write('msqe ' + str(epoch + 1) + ' quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()


    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    else:
        q = q / torch.pow(original_data)

    return quant_data, q


def quantize(data, bits, args, write, key, epoch, seed):
    np.random.seed(seed)
    random.seed(seed)
    if args.quant_function == 'msqe' :
        return quantize_msqe_liu(data, bits, args.q_cal, write, key, epoch)
        # return quantize_msqe(data, bits, args.q_cal, write, key, epoch)


