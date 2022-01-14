# 919106840212
# 周运莲
# 机器学习第三次作业
# 感知机
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

fileName1 = "./dataset/ex4x.dat"
fileName2 = "./dataset/ex4y.dat"


# 加载数据
def load_data(fileName1, fileName2):
    train_X = np.loadtxt(fileName1)
    train_Y = np.loadtxt(fileName2)
    dataTot = train_X.shape[0]
    return dataTot, train_X, train_Y


# 把特征数据进行标准化为均匀分布,并同时对数据进行初步处理,
def data_process(train_X):
    max_X = train_X.max(axis=0)  # 取每一列的最大值
    min_X = train_X.min(axis=0)  # 取每一列的最小值
    train_X = (train_X - min_X) / (max_X - min_X)
    # 插入x0=1这一列
    train_X = np.insert(train_X, 0, values=1.0, axis=1)
    return train_X.T


# 损失函数计算
def Calculate_Loss(H_Y, dataTot, train_X, train_Y, theta_array):
    lossValue = 0.0
    for i in range(dataTot):
        lossValue += (H_Y[i] - train_Y[i]) * (theta_array.dot(train_X[:, i]))
    return lossValue


# 假设函数；根据给定某一个样本x_array 预测分类结果
def H(x_array, theta_array):
    tempH = theta_array.dot(x_array)
    if type(tempH) == np.ndarray:
        kind = []
        for i in range(tempH.size):
            if tempH[i] >= 0:
                kind.append(1)
            else:
                kind.append(0)
    else:
        if tempH >= 0:
            kind = 1
        else:
            kind = 0
    return kind


# H_Y: 预测值，train_Y真实值  一维；计算预测准确率 预测正确的数量/总数
def Calculate_Accuracy(H_Y, train_Y):
    rightCnt = 0  # 预测值 正确的数量
    totalNum = train_Y.size  # totalNum
    for i in range(totalNum):
        if H_Y[i] == train_Y[i]:
            rightCnt += 1
    return rightCnt / totalNum


# 感知机
def Perceptron_SGD(train_X, learning_rate, dataTot, train_Y):
    theta_array = np.zeros((train_X.shape[0],))
    plt.ion()
    i = 0
    while 1:
        randomIndex = random.randint(0, dataTot - 1)
        ef = train_Y[randomIndex] - H(train_X[:, randomIndex], theta_array) * train_X[:, randomIndex]
        theta_array += learning_rate * ef
        temp_accuracy = print_Pic(theta_array, i, train_X, train_Y, dataTot)
        i += 1
        plt.pause(0.01)
        if temp_accuracy > 0.8:  # 当准确率大于0.8的时候停止
            break
        plt.cla()
    plt.ioff()
    plt.show()


# 作图
def print_Pic(theta_array, i, train_X, train_Y, dataTot):
    mk = ['o', '^'];
    cs = ['g', 'r', ]
    classAns = H(train_X, theta_array)
    loss_temp = Calculate_Loss(classAns, dataTot, train_X, train_Y, theta_array)
    print("============第 %d 轮===========" % (i))
    print("当前损失值为： %8.f" % (loss_temp))
    tempAccuracy = Calculate_Accuracy(classAns, train_Y)
    print("当前的准确率为：%.8f" % (tempAccuracy))
    plt.xlabel('x1')
    plt.ylabel('x2')
    for j in range(dataTot):
        plt.scatter(train_X[1, j], train_X[2, j],
                    marker=mk[int(train_Y[j])],
                    c=cs[int(train_Y[j])])
    x = np.linspace(0, 1, 50)
    y = -(theta_array[0] + theta_array[1] * x) / theta_array[2]  # 分类线: x2 = -(w0 + w1*x1)/w2
    plt.plot(x, y, 'k')
    plt.suptitle('Perceptron——SGD')
    return tempAccuracy


if __name__ == "__main__":
    dataTot, train_X, train_Y = load_data(fileName1, fileName2)
    train_X = data_process(train_X)
    Perceptron_SGD(train_X, 0.01, dataTot, train_Y)
