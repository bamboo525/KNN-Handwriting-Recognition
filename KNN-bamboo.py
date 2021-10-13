# -*- coding:utf-8 -*-
"""
作者：86173
日期：2021年09月27日
"""

import random
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
import numpy as np
from math import sqrt

g_dataset = {} # dataset数据集
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32 # 行数
NUM_COLS = 32 # 列数
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'

# kNN parameter 参数
KNN_NEIGHBOR = 7


##########################
##### Load Data  #########
##########################

# Convert next digit from input file as a vector 将输入文件中的下一个数字转换为矢量
# Return (digit, vector) or (-1, '') on end of file 在文件末尾返回（数字,向量）或(-1, '')
def read_digit(p_fp):
    # read entire digit (inlude linefeeds) 读取整个数字（包括换行符）
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1, bits
    # convert bit string as digit vector 将位字符串转换为数字向量
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val, vec


# Parse all digits from training file 解析训练文件中的所有数字
# and store all digits (as vectors) 并存储所有数字（作为向量）
# in dictionary g_dataset 字典中的g_dataset
def load_data(p_filename=DATA_TRAINING):
    global g_dataset
    # Initial each key as empty list 每个键的首字母都是空列表
    g_dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            g_dataset[val].append(vec)


##########################
##### kNN Models #########
##########################

# Given a digit vector, returns 给定一个数字向量，返回
# the k nearest neighbor by vector distance 向量距离的k近邻
def knn(p_v, size=KNN_NEIGHBOR):
    nn = []
    for d, vectors in g_dataset.items():
        for v in vectors:
            dist = round(distance(p_v, v), 2)
            nn.append((dist, d))

    # TODO: find the nearest neigbhors 找到最近的邻居
    # 对列表nn进行排序(含有元组的列表排序方法)
    sortednn = sorted(nn, key=lambda x: x[0])
    return sortednn[0:size] # 取出前size个


# Based on the knn Model (nearest neighhor) 基于knn模型（最近的Neighor）
# return the target value 返回目标值
# 基于knn模型（最近的Neighor），返回目标值
def knn_by_most_common(p_v):
    nn = knn(p_v)

    # TODO: target value 目标值

    """找出nn元组列表中出现次数最多的value
            键：值 （key：value)"""

    # collection_nn = Counter(nn) # Counter类排序
    d_nn = dict(nn) # 将元组列表转为字典
    target_key = max(d_nn.keys()) # 寻找最大的key
    # return target_value
    target = d_nn[target_key] # 输出最大key对应的value
    return target

##########################
##### Prediction  ########
##########################

# Make prediction based on kNN model 基于kNN模型进行预测
# Parse each digit from the predict file 解析predict文件中的每个数字
# and print the predicted balue 并打印预测的balue
def predict(p_filename=DATA_PREDICT):
    # TODO
    print('TO DO: show results of prediction')

    # load_data(p_filename)
    # 解析predict文件中的每个数字
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            result = knn_by_most_common(vec) # 调用knn算法进行测试
            print(result)

##########################
#####  Accuracy  #########
##########################

# Compile an accuracy report by 编制一份准确度报告
# comparing the data set with every 将数据集与每个
# digit from the testing file 测试文件中的数字
def validate(p_filename=DATA_TESTING):
    global g_test_bad, g_test_good
    g_test_bad = defaultdict(int)
    g_test_good = defaultdict(int)

    data_by_random(100) # 为每个数据随机选择100个样本
    load_data(p_filename) # 读取数据
    list1 = []
    for v in g_dataset.values():
        a = len(v)
        list1.append(a)
    l = sum(list1) # 获取数据g_dataset.values的总值，作为进度百分号的分母
    i = 0 # 计算测试数据次数
    print('TODO: Training Info')
    for val, vecs in g_dataset.items():
        for vec in vecs:
            test_result = knn_by_most_common(vec)
            if test_result == val:
                g_test_good[val] += 1
            else:
                g_test_bad[val] += 1
            i += 1
            print("\r进度：{}%".format(round(100 * i / l, 2)), end='')
    print()

    # 法二：【此方法样本数较小例如10，判断速度快，同时准确率也会降低，样本为100时基本无差别】
    # data_by_random(10)  # 为每个数据随机选择10个样本
    # g_data = defaultdict(list)
    # with open(p_filename) as f:
    #     while True:
    #         val, vec = read_digit(f)
    #         if val == -1:
    #             break
    #         g_data[val].append(vec)
    #     l = sum([len(v) for v in g_data.values()]) # 五行的简洁式
    #
    #     i = 0 # 计算测试数据次数
    #     print('TODO: Training Info')
    #     for val, vecs in g_data.items():
    #         for vec in vecs:
    #             test_result = knn_by_most_common(vec)
    #             if test_result == val:
    #                 g_test_good[val] += 1
    #             else:
    #                 g_test_bad[val] += 1
    #             i += 1
    #             print("\r进度：{}%".format(round(100 * i / l, 2)), end='')
    #     print()

    start = datetime.now()

    # TODO: Validate your kNN model with 使用验证kNN模型
    # digits from test file. 数据来自测试文件

    stop = datetime.now()
    show_test(start, stop)


##########################
##### Data Models ########
##########################

# Randomly select X samples for each digit 为每个数据随机选择X个样本
def data_by_random(size=25):
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit], size)


##########################
##### Vector     #########
##########################

# Return distance between vectors v & w  向量 v & w 之间的返回距离
def distance(v, w):

    """ 欧氏距离计算 """

    vector1 = np.array(v)
    vector2 = np.array(w)
    op = np.linalg.norm(vector1 - vector2) # 欧氏距离计算
    return op


##########################
##### Report     #########
##########################

# Show info for training data set 显示训练数据集的信息
def show_info():
    print('TODO: Training Info')
    for d in range(10):
        print(d, '=', len(g_dataset[d]))


# Show test results 显示测试结果
def show_test(start="????", stop="????"):
    print('Beginning of Validation @ ', start) # 验证的开始@
    print('TODO: Testing Info')

    """
    format格式化字符串函数 
    round()方法返回浮点数x的四舍五入值,逗号后面表示保留几位小数
     """

    # result = ['KNN_NEIGHBOR={}\n'.format(KNN_NEIGHBOR)]
    print('各数字准确率为：')
    sum = 0
    for d in range(10):
        good = g_test_good[d]
        bad = g_test_bad[d]
        accuracy = 100 * good / (good + bad)  # 准确率
        # format格式化字符串函数
        # output = "{}={} {} {}%\n".format(d, good, bad, round(accuracy, 2))
        # result.append(output)
        sum += accuracy
        print(str(d) + '=' + str(good) + ' ' + str(bad) + ' ' + str(round(accuracy, 2)) + '%')
    average = "average:{}%\n".format(round(sum / 10, 2))
    print(average)
    print('End of Validation @ ', stop)  # 验证结束@
    # result.append(average)
    # with open('准确率and平均值.txt', 'a') as f:
    #     f.writelines(result)
    # 所有注释部分可把结果输出到txt文件中

if __name__ == '__main__':
    load_data()
    show_info()
    validate()
    predict()
