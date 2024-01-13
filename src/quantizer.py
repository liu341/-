from __future__ import print_function
import torch
from options import args_parser
import copy
import numpy as np
import math
import random

args = args_parser()


def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)


def sum_power(bits):
    m1 = 0
    m2 = 0
    if bits == 3:
        m1 = 1
        m2 = 1
    elif bits == 5:
        m1 = 3
        m2 = 1
    elif bits == 4:
        m1 = 2
        m2 = 1
    q1 = [0.]
    q2 = [0.]
    for i in range(2 ** m1 - 1):
        q1.append(1/(2 ** (2**m1 - (i+1))))
    for i in range(2 ** m2 - 1):
        q2.append(1/(2 ** (2**m2 - (i+1))))
    value = []
    for d1 in q1:
        for d2 in q2:
            value.append(d1+d2)
    value = np.array(sorted(set(value)))
    value_ = value * (-1)
    bound = np.concatenate((value,value_))
    bound = np.array(sorted(set(list(bound))))
    return bound


# sum_power(5)
def quantize_SP2(args, data, bits, q_cal, write, key, epoch):
    data_shape = data.shape
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    original_data = copy.deepcopy(data)
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    q = 0
    n_clusters = pow(2, bits) - 1
    if weight.shape[0] > n_clusters:
        bound = sum_power(bits)
        max_ = max(abs(weight))
        bound = max_ * bound
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight_SP2.txt', 'a')
            f.write(str(epoch + 1) + '\n' + 'bound ' + 'SP2' + '\n' + str(bound.tolist()) + '\n' + str(
                key) + '\n' + str(weight.reshape([1, -1])) + '\n')
        for k in range(1, len(bound)):
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
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f.write('SP2 ' + str(epoch + 1) + '\n' + 'quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    # print('quant_data',quant_data)
    return quant_data, q


def quantize_LQ(args, data, bits, q_cal, write, key, epoch):
    n_clusters = pow(2, bits) - 1  # 7
    data_shape = data.shape
    # np.set_printoptions(suppress=True)
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    original_data = copy.deepcopy(data)
    q = 0
    if weight.shape[0] >= n_clusters:
        if write == 1:
            f = open('nnq__.txt', 'a')
            f.write(
                'bound ' + 'LQ' + '\n' + str(weight.reshape([1, -1])) + '\n')
        input_vector = weight.reshape([1, -1])
        sgn = np.sign(input_vector[0])
        max_input = max(abs(input_vector[0]))
        weight_temp = input_vector[0] / max_input
        input_0 = np.where((weight_temp == 0))
        idx_index = [i for i in range(len(weight_temp))]
        input_no_0 = np.array(list(set(idx_index) - set(input_0[0])))
        x_temp = np.log2(abs(weight_temp[input_no_0])).round()
        min_ = 1 - 2 ** (bits - 1)
        max_ = 1
        x_temp[x_temp <= min_] = 0
        x_temp[x_temp >= max_] = max_ - 1
        weight_temp[input_no_0] = 2 ** x_temp
        weight_temp[input_0] = 0
        input_vector[0] = weight_temp * max_input * sgn
        weight = input_vector.reshape(weight.shape)
        if write == 1:
            f.write('LQ ' + ' quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    return quant_data, q


def quantize_LCQ(args, data, bits, q_cal, b, key, epoch):
    n_clusters = pow(2, bits) - 1  # 7
    data_shape = data.shape
    # np.set_printoptions(suppress=True)
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    original_data = copy.deepcopy(data)
    q = 0
    if n_clusters < weight.shape[0] < 1000:
        bound = np.zeros(n_clusters + 1)
        # # if epoch == 0:
        min_ = torch.min(data)
        max_ = torch.max(data)
        bound[0] = min_
        bound[n_clusters] = max_
        for i in range(1, n_clusters):
            bound[i] = min_ + i * (max_ - min_) / n_clusters

        bound_msqe = np.array(bound)
        for k in range(1, n_clusters + 1):
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
    elif weight.shape[0] >= 1000:
        write = 0
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('LCQ.txt', 'a')
            f.write(
                'bound ' + 'LCQ' + '\n' + str(weight.reshape([1, -1])) + '\n')
        input_vector = weight.reshape([1, -1])

        sgn = np.sign(input_vector[0])
        # max_input = max(abs(input_vector[0]))
        max_input = 1
        weight_temp = abs(input_vector[0]) / max_input

        K = n_clusters
        bound = np.zeros(K)
        bound[0] = np.min(weight_temp)
        bound[K - 1] = np.max(weight_temp)
        for i in range(1, K - 1):
            bound[i] = bound[0] + i * (bound[K - 1] - bound[0]) / (K - 1)
        bound_k = np.exp(bound) / sum(np.exp(bound))
        r_k = bound_k * K
        d_k = np.zeros(K + 1)
        b_k = np.zeros(K + 1)
        for i in range(1, len(d_k)):
            d_k[i] = i / K
        for i in range(1, len(b_k)):
            b_k[i] = sum(bound_k[0:i])
        d_k[0] = 0
        b_k[0] = 0
        for k in range(1, len(d_k)):
            v_idx = np.where((weight_temp >= d_k[k - 1]) & (weight_temp <= d_k[k]))
            v = weight_temp[v_idx[0]]
            weight_temp[v_idx[0]] = r_k[k - 1] * (v - d_k[k - 1]) + b_k[k - 1]

        # s = 2 ** (bits-1) - 1
        # weight_temp = (weight_temp * s).round()
        # weight_temp /= s
        bound_v = np.zeros(n_clusters + 1)
        # # if epoch == 0:
        min_ = np.min(weight_temp)
        max_ = np.max(weight_temp)
        bound_v[0] = min_
        bound_v[n_clusters] = max_
        for i in range(1, n_clusters):
            bound_v[i] = min_ + i * (max_ - min_) / n_clusters
        for k in range(1, n_clusters + 1):
            pa = np.where((weight_temp >= bound_v[k - 1]) & (weight_temp < bound_v[k]))
            tmp = weight_temp[pa[0]]
            k_max = bound_v[k]
            k_min = bound_v[k - 1]
            if k_min != k_max:
                p_min = (k_max - tmp) / (k_max - k_min)
            else:
                p_min = 1
            prob = np.random.rand(tmp.shape[0])  # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
            tmp[prob < p_min] = k_min  # 随机量化
            tmp[prob >= p_min] = k_max
            weight_temp[pa[0]] = tmp
        for k in range(1, len(b_k)):
            v_idx = np.where((weight_temp >= b_k[k - 1]) & (weight_temp <= b_k[k]))
            v = weight_temp[v_idx[0]]
            weight_temp[v_idx[0]] = (v - b_k[k - 1]) / r_k[k - 1] + d_k[k - 1]
        input_vector[0] = weight_temp * max_input * sgn
        weight = input_vector.reshape(weight.shape)
        if write == 1:
            f.write('LCQ ' + ' quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    return quant_data, q


def build_power_value(B=3, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:  # k = 2,
        if B == 2:
            for i in range(3):  # 0 1 2
                base_a.append(2 ** (-i - 1))  # -1 -2 -3
        elif B == 4:
            for i in range(3):
                base_a.append(2 ** (-2 * i - 1))  # -1 -3 -5
                base_b.append(2 ** (-2 * i - 2))  # -2 -4 -6
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))  # -1 -4 -7
                base_b.append(2 ** (-3 * i - 2))  # -2 -5 -8
                base_c.append(2 ** (-3 * i - 3))  # -3 -6 -9
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))  # -1 -2
                else:
                    base_b.append(2 ** (-i - 1))  # -3
                    base_a.append(2 ** (-i - 2))  # -1 -2 -4
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))  # -1 -3 -5
                    base_b.append(2 ** (-2 * i - 2))  # -2 -4 -6
                else:
                    base_c.append(2 ** (-2 * i - 1))  # -5
                    base_a.append(2 ** (-2 * i - 2))  # -1 -3 -5 -6
                    base_b.append(2 ** (-2 * i - 3))  # -2 -4 -6 -7
        else:
            pass
    else:
        for i in range(2 ** B - 1):
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    # values[0] = 2 ** (-B - 2)
    values = sorted(np.array(values))
    # print(values)
    # values_ = []
    #
    # for i in range(len(values)):
    #     if i >= (2 ** B) / 2:
    #         break
    #     v = 0
    #     for j in range(i, int(i + (2 ** B) / 2 + 1)):
    #         v += values[j]
    #     if i < (2 ** B) / 2:
    #         values_.append(v / int((2 ** B) / 2 + 1))
    # print(values_)
    return values


def quantize_APoT(args, data, bits, q_cal, b, key, epoch):
    if args.multiply == True:
        num = 1000
    else:
        num = 10000000000
    data_shape = data.shape
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    original_data = copy.deepcopy(data)
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    q = 0
    bound = []
    n_clusters = pow(2, bits) - 1
    if n_clusters < weight.shape[0] < 1000:
        bound = np.zeros(n_clusters + 1)
        # # if epoch == 0:
        min_ = torch.min(data)
        max_ = torch.max(data)
        bound[0] = min_
        bound[n_clusters] = max_
        for i in range(1, n_clusters):
            bound[i] = min_ + i * (max_ - min_) / n_clusters

        bound_msqe = np.array(bound)
        for k in range(1, n_clusters + 1):
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
    elif len(weight) > 1000:
        bound = build_power_value(bits - 1, True)
        bound_ = np.array(bound) * (-1)
        bound = np.hstack((bound,bound_))
        bound = np.array(sorted(set(list(bound))))
        max_ = max(abs(weight))
        # min_ = min(abs(weight))
        # scale = max_ / max(bound)
        bound = max_ * bound
        # bound[0] = bound[0] + min_

        # weight_sign = np.sign(weight)
        # weight_abs = np.abs(weight)
        if b == 1:
            # bound_temp = copy.deepcopy(bound)
            # bound_temp = bound_temp.tolist()
            # for i in range(1, len(bound_temp)):
            #     bound_temp.append(-bound_temp[i])
            # bound_temp = sorted(np.array(bound_temp))
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write(
                str(epoch + 1) + '\n' + 'bound ' + 'APoT' + '\n' + str(bound) + '\n' + str(key) + '\n' + str(
                    weight.reshape([1, -1])) + '\n')
        for k in range(1, n_clusters):
            pa = np.where((weight >= bound[k - 1]) & (weight <= bound[k]))
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
        if b == 1:
            np.set_printoptions(threshold=100000000000)
            f.write('APoT ' + str(epoch + 1) + '\n' + 'quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    # print('quant_data',quant_data)
    return quant_data, q


def first_bound(data, l, a0, a1):
    index_l_a0 = np.where((data >= l) & (data <= a0))
    index_a0_a1 = np.where((data >= a0) & (data <= a1))
    num_l_a0 = len(index_l_a0[0])
    num_a0_a1 = len(index_a0_a1[0])
    sum_l_a0 = sum(data[index_l_a0[0]])
    sum_a0_a1 = sum(data[index_a0_a1[0]])
    dic = 2 * a0 * num_l_a0 - 2 * sum_l_a0 + sum_a0_a1 - a1 * num_a0_a1
    return dic


def last_bound(data, r, an_1, an_2):
    index_an_2_an_1 = np.where((data >= an_2) & (data <= an_1))
    index_an_1_r = np.where((data >= an_1) & (data <= r))
    num_an_2_an_1 = len(index_an_2_an_1)
    num_an_1_r = len(index_an_1_r)
    sum_an_2_an_1 = sum(data[index_an_2_an_1])
    sum_an_1_r = sum(data[index_an_1_r])
    dic = 2 * an_1 * num_an_1_r - 2 * sum_an_1_r + sum_an_2_an_1 - an_2 * num_an_2_an_1
    return dic


def binary_search(array, input_vector, bound_msqe, n_clu, t, c_dic=10000000, d_dic=0):
    """二分查找法递归实现"""
    if len(array) == 0:
        return d_dic
    mid_index = len(array) // 2
    if t:
        dic = last_bound(input_vector, bound_msqe[n_clu], input_vector[array[mid_index]], bound_msqe[n_clu - 1])
    else:
        dic = first_bound(input_vector, bound_msqe[0], input_vector[array[mid_index]], bound_msqe[1])
    if abs(dic) < abs(c_dic):
        c_dic = dic
        d_dic = array[mid_index]
    # if array[mid_index] == data:
    #     return True
    return binary_search(array[mid_index + 1:], input_vector, bound_msqe, n_clu, t, c_dic, d_dic) \
        if dic < 0 else \
        binary_search(array[:mid_index], input_vector, bound_msqe, n_clu, t, c_dic, d_dic)


def weight_normal(weight):
    p = np.mean(weight)
    sigma = np.mean((weight - p) ** 2)
    kesi = 0.00001
    weight_nor = (weight - p) / (sigma + kesi)
    return weight_nor


def clipping_APoT(args, data, bits, q_cal, b, key, epoch):
    if args.multiply == True:
        num = 1000
    else:
        num = 10000000000
    data_shape = data.shape
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    original_data = copy.deepcopy(data)
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    q = 0
    bound = []
    n_clusters = pow(2, bits)-1
    if n_clusters < weight.shape[0] < 1000:
        min = torch.min(data)
        max = torch.max(data)
        bound = np.zeros(n_clusters + 1)
        bound[0] = min
        bound[n_clusters] = max
        for i in range(1, n_clusters):
            bound[i] = min + i * (max - min) / n_clusters

        weight_temp = weight.reshape([1, -1])
        input_vector = sorted(weight_temp[0])
        input_vector = np.array(input_vector)
        index_l_a1 = np.where((input_vector >= bound[0]) & (input_vector <= bound[1]))
        index_an_2_r = np.where(
            (input_vector >= bound[n_clusters - 1]) & (input_vector <= bound[n_clusters]))

        a_index = binary_search(index_l_a1[0], input_vector, bound, n_clu=n_clusters, t=False)
        b_index = binary_search(index_an_2_r[0], input_vector, bound, n_clu=n_clusters, t=True)
        bound[0] = input_vector[a_index]
        bound[n_clusters] = input_vector[b_index]
        min_ = bound[0]
        max_ = bound[n_clusters]
        for i in range(1, n_clusters):
            bound[i] = min_ + i * (max_ - min_) / n_clusters
        for k in range(1, n_clusters + 1):
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
    elif len(weight) > 1000:
        min_w = np.min(weight)
        max_w = np.max(weight)
        bound = np.zeros(n_clusters + 1)
        bound[0] = min_w
        bound[n_clusters] = max_w
        for i in range(1, n_clusters):
            bound[i] = min_w + i * (max_w - min_w) / n_clusters

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

        weight_temp = weight.reshape([1, -1])
        input_vector = np.array(sorted(weight_temp[0]))
        index_l_a1 = np.where((input_vector >= bound[0]) & (input_vector <= bound[1]))
        index_an_2_r = np.where(
            (input_vector >= bound[n_clusters - 1]) & (input_vector <= bound[n_clusters]))
        a_index = binary_search(index_l_a1[0], input_vector, bound, n_clu=n_clusters, t=False)
        b_index = binary_search(index_an_2_r[0], input_vector, bound, n_clu=n_clusters, t=True)
        # max_ = 0
        if abs(input_vector[a_index]) <= abs(input_vector[b_index]):
            max_ = abs(input_vector[b_index])
        else:
            max_ = abs(input_vector[a_index])
        bound = build_power_value(bits - 1, True)
        bound_ = np.array(bound) * (-1)
        bound = np.hstack((bound, bound_))
        bound = np.array(sorted(set(list(bound))))
        # min_ = min(abs(weight))
        # scale = max_ / max(bound)
        bound = max_ * bound
        # bound[0] = bound[0] + min_

        # weight_sign = np.sign(weight)
        # weight_abs = np.abs(weight)
        if b == 1:
            # bound_temp = copy.deepcopy(bound)
            # bound_temp = bound_temp.tolist()
            # for i in range(1, len(bound_temp)):
            #     bound_temp.append(-bound_temp[i])
            # bound_temp = sorted(np.array(bound_temp))
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write(
                str(epoch + 1) + '\n' + 'bound ' + 'APoT' + '\n' + str(bound) + '\n' + str(key) + '\n' + str(
                    weight.reshape([1, -1])) + '\n')
        for k in range(1, n_clusters):
            pa = np.where((weight >= bound[k - 1]) & (weight <= bound[k]))
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
        pa0 = np.where((weight < bound[0]))
        pa1 = np.where((weight > bound[n_clusters-1]))
        weight[pa0[0]] = -1*max_
        weight[pa1[0]] = max_
        if b == 1:
            np.set_printoptions(threshold=100000000000)
            f.write('clipping_APoT ' + str(epoch + 1) + '\n' + 'quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    # print('quant_data',quant_data)
    return quant_data, q

def clipping_uniformity(args, data, bits, q_cal, write, key, epoch):
    n_clusters = pow(2, bits) - 1
    if args.multiply == True:
        num = 1000
    else:
        num = 10000000000
    data_shape = data.shape
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    original_data = copy.deepcopy(data)
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    q = 0
    # weight = weight_normal(weight)
    if weight.shape[0] > n_clusters:
        min_w = np.min(weight)
        max_w = np.max(weight)
        bound = np.zeros(n_clusters + 1)
        bound[0] = min_w
        bound[n_clusters] = max_w
        for i in range(1, n_clusters):
            bound[i] = min_w + i * (max_w - min_w) / n_clusters

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

        weight_temp = weight.reshape([1, -1])
        input_vector = np.array(sorted(weight_temp[0]))
        index_l_a1 = np.where((input_vector >= bound[0]) & (input_vector <= bound[1]))
        index_an_2_r = np.where(
            (input_vector >= bound[n_clusters - 1]) & (input_vector <= bound[n_clusters]))
        # if first_bound(input_vector,bound_msqe[0],index_l_a1[len(index_l_a1)],)
        # for i in range(len(index_an_2_an_1[0])):
        #     first_b = last_bound(input_vector,bound_msqe[7],input_vector[index_an_2_an_1[0][i]],bound_msqe[6])
        #     dic.append(first_b)

        a_index = binary_search(index_l_a1[0], input_vector, bound, n_clu=n_clusters, t=False)
        b_index = binary_search(index_an_2_r[0], input_vector, bound, n_clu=n_clusters, t=True)
        bound[0] = input_vector[a_index]
        bound[n_clusters] = input_vector[b_index]
        min_ = bound[0]
        max_ = bound[n_clusters]
        for h in range(1, n_clusters):
            bound[h] = min_ + h * (max_ - min_) / n_clusters
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write(str(epoch + 1) + '\n' + 'bound ' + 'clipping_uniform' + '\n' + str(bound.tolist()) + '\n' + str(
                key) + '\n' + str(weight.reshape([1, -1])) + '\n')
        for k in range(1, n_clusters + 1):
            pa = np.where((weight >= bound[k - 1]) & (weight <= bound[k]))
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
        # if weight.shape[0] >= 1000:
        pa0 = np.where((weight < bound[0]))
        pa1 = np.where((weight > bound[n_clusters]))
        weight[pa0[0]] = bound[0]
        weight[pa1[0]] = bound[n_clusters]
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f.write('uniform ' + str(epoch + 1) + '\n' + 'quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    # print('quant_data',quant_data)
    return quant_data, q


def quantize_uniformity(args, data, bits, q_cal, write, key, epoch):
    n_clusters = pow(2, bits) - 1
    if args.multiply == True:
        num = 1000
    else:
        num = 10000000000
    data_shape = data.shape
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    original_data = copy.deepcopy(data)
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    q = 0
    # weight = weight_normal(weight)
    if weight.shape[0] > n_clusters:
        min_w = np.min(weight)
        max_w = np.max(weight)
        bound = np.zeros(n_clusters + 1)
        bound[0] = min_w
        bound[n_clusters] = max_w
        for i in range(1, n_clusters):
            bound[i] = min_w + i * (max_w - min_w) / n_clusters

        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write(str(epoch + 1) + '\n' + 'bound ' + 'uniform' + '\n' + str(bound.tolist()) + '\n' + str(
                key) + '\n' + str(weight.reshape([1, -1])) + '\n')
        for k in range(1, n_clusters + 1):
            pa = np.where((weight >= bound[k - 1]) & (weight <= bound[k]))
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
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f.write('uniform ' + str(epoch + 1) + '\n' + 'quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    # print('quant_data',quant_data)
    return quant_data, q


def quantize_msqe(data, bits, q_cal, write, key, epoch):  # data, 3, SQE
    #fff = open('One_note/running.txt', 'a')
    #fff.write('bits:{}\n'.format(bits))
    # 设置 NumPy 打印选项，确保打印所有元素
    # np.set_printoptions(threshold=np.inf)
    #fff.write(f'data_{epoch+1}:\n{data.cpu().numpy()}\n')
    #fff.close()
    n_clusters = pow(2, bits) - 1  # 7
    data_shape = data.shape
    # np.set_printoptions(suppress=True)
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    original_data = copy.deepcopy(data)
    q = 0
    bound_msqe = []
    original_bound = []
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
            f = open('One_note/{}weight.txt'.format(epoch+1), 'a')
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
            f = open('One_note/{}weight.txt'.format(epoch), 'a')
            f.write('msqe ' + str(epoch + 1) + ' quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()


    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    else:
        q = q / torch.pow(original_data)

    return quant_data, q


def clipping_msqe(data, bits, q_cal, write, key, epoch):
    n_clusters = pow(2, bits) - 1  # 7
    data_shape = data.shape
    # np.set_printoptions(suppress=True)
    weight = copy.deepcopy(data.cpu().numpy().reshape([-1, 1]))
    if (weight == 0).all() or (weight == 1).all():
        return data, 0
    original_data = copy.deepcopy(data)
    q = 0
    bound_msqe = []
    # weight = weight_normal(weight)
    if n_clusters < weight.shape[0] < 1000:
        min = torch.min(data)
        max = torch.max(data)
        bound = np.zeros(n_clusters + 1)
        bound[0] = min
        bound[n_clusters] = max
        for i in range(1, n_clusters):
            bound[i] = min + i * (max - min) / n_clusters

        weight_temp = weight.reshape([1, -1])
        input_vector = sorted(weight_temp[0])
        input_vector = np.array(input_vector)
        index_l_a1 = np.where((input_vector >= bound[0]) & (input_vector <= bound[1]))
        index_an_2_r = np.where(
            (input_vector >= bound[n_clusters - 1]) & (input_vector <= bound[n_clusters]))

        a_index = binary_search(index_l_a1[0], input_vector, bound, n_clu=n_clusters, t=False)
        b_index = binary_search(index_an_2_r[0], input_vector, bound, n_clu=n_clusters, t=True)
        bound[0] = input_vector[a_index]
        bound[n_clusters] = input_vector[b_index]
        min_ = bound[0]
        max_ = bound[n_clusters]
        for i in range(1, n_clusters):
            bound[i] = min_ + i * (max_ - min_) / n_clusters
        for k in range(1, n_clusters + 1):
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
    if weight.shape[0] >= 1000:  # weight.shape => (row,cal)
        bound = np.zeros(n_clusters + 1)
        min = torch.min(data)
        max = torch.max(data)
        bound[0] = min
        bound[n_clusters] = max
        for i in range(1, n_clusters):
            bound[i] = min + i * (max - min) / n_clusters

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
        if write == 1:
            np.set_printoptions(threshold=100000000000)
            f = open('One_note/weight.txt', 'a')
            f.write('bound ' + 'clipping_msqe' + '\n' +
                    str(weight.reshape([1, -1])) + '\n')
        bound_msqe = copy.deepcopy(np.array(sorted(bound)))
        weight_temp = weight.reshape([1, -1])
        # print((weight_temp == 0).all())
        np.set_printoptions(suppress=True)
        input_vector = sorted(weight_temp[0])
        input_vector = np.array(input_vector)
        # dic = []
        index_l_a1 = np.where((input_vector >= bound_msqe[0]) & (input_vector <= bound_msqe[1]))
        index_an_2_r = np.where(
            (input_vector >= bound_msqe[n_clusters - 1]) & (input_vector <= bound_msqe[n_clusters]))
        # if first_bound(input_vector,bound_msqe[0],index_l_a1[len(index_l_a1)],)
        # for i in range(len(index_an_2_an_1[0])):
        #     first_b = last_bound(input_vector,bound_msqe[7],input_vector[index_an_2_an_1[0][i]],bound_msqe[6])
        #     dic.append(first_b)

        a_index = binary_search(index_l_a1[0], input_vector, bound_msqe, n_clu=n_clusters, t=False)
        b_index = binary_search(index_an_2_r[0], input_vector, bound_msqe, n_clu=n_clusters, t=True)
        bound_msqe[0] = input_vector[a_index]
        bound_msqe[n_clusters] = input_vector[b_index]
        for iter in range(1000000):
            bound_temp = copy.deepcopy(bound_msqe)
            for i in range(1, n_clusters):
                # temp1 = input_vector - bound_msqe[i - 1]
                # temp2 = input_vector - bound_msqe[i + 1]
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

        # if re:
        #     bound_msqe = np.delete(bound_msqe, [0, 9])
        for k in range(1, n_clusters + 1):
            # temp1 = weight - bound_msqe[k - 1]
            # temp2 = weight - bound_msqe[k]
            # pa = np.where((temp1 * temp2) <= 0)
            pa = np.where((weight >= bound_msqe[k - 1]) & (weight <= bound_msqe[k]))
            tmp = weight[pa[0]].reshape([-1, 1])
            k_max = bound_msqe[k]
            k_min = bound_msqe[k - 1]
            if k_min != k_max:
                p_min = (k_max - tmp) / (k_max - k_min)
            else:
                p_min = 1
            # np.random.seed(1)
            prob = np.random.rand(tmp.shape[0], tmp.shape[1])  # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
            tmp[prob < p_min] = k_min  # 随机量化
            tmp[prob >= p_min] = k_max
            weight[pa[0]] = tmp
        np.set_printoptions(threshold=100000000000)
        pa0 = np.where((weight < bound_msqe[0]))
        pa1 = np.where((weight > bound_msqe[n_clusters]))
        weight[pa0[0]] = bound_msqe[0]
        weight[pa1[0]] = bound_msqe[n_clusters]
        if write == 1:
            f.write('clipping_msqe ' + ' quant data' + '\n' + str(weight.reshape([1, -1])) + '\n')
            f.close()
        # if q_cal == 'expectation':
        #     q = torch.max(q, torch.max(torch.tensor(np.power(original_tmp - tmp, 2) / original_tmp)))
    quant_data = torch.tensor(weight.reshape(data_shape)).cuda()
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(original_data - quant_data, 2))
    else:
        q = q / torch.pow(original_data)
    return quant_data, q


def quantize_LSQ(args, data, bits, q_cal='SQE'):
    lsq_quan = LsqQuan(bits)
    x = lsq_quan(data)
    if q_cal == 'SQE':
        q = torch.sum(torch.pow(data - x, 2)).cuda()
    return x,q

def quantize(data, bits, args, write, key, epoch, seed):
    np.random.seed(seed)
    random.seed(seed)
    if args.quant_function == 'uniformity':
        return quantize_uniformity(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'clipping_uniformity':
        return clipping_uniformity(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'msqe':
        return quantize_msqe(data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'clipping_msqe':
        return clipping_msqe(data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'APoT':
        return quantize_APoT(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'clipping_APoT':
        return clipping_APoT(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'SP2':
        return quantize_SP2(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'LQ':
        return quantize_LQ(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'LCQ':
        return quantize_LCQ(args, data, bits, args.q_cal, write, key, epoch)
    elif args.quant_function == 'LSQ':
        return quantize_LSQ(args, data, bits, args.q_cal)

