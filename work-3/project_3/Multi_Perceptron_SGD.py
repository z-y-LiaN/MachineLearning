# 919106840212
# 周运莲
# 机器学习第三次作业
# 多分类感知机
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


# H_Y: 预测值，train_Y真实值  一维；计算预测准确率 预测正确的数量/总数
def Calculate_Accuracy(H_Y, train_Y):
    rightCnt = 0  # 预测值 正确的数量
    totalNum = train_Y.size  # totalNum
    for i in range(totalNum):
        if H_Y[i] == train_Y[i]:
            rightCnt += 1
    return rightCnt / totalNum

#  损失函数计算
def Calculate_Loss(dataTot, train_X, train_Y, theta_array):
    lossValue = 0.
    preH = H(train_X, theta_array, classNum)[0]
    for i in range(dataTot):
        x = train_X[:, i]
        lossValue += preH[i] - theta_array[:, int(train_Y[i])].dot(x)
    return lossValue

#  假设函数；根据给定某一个样本x_array 预测分类结果
def H(x_array, theta_array, classNum):
    # 得到数据个数
    if x_array.ndim == 1:
        dataNum = 1
    else:
        dataNum = x_array.shape[1]
    tempResult = np.zeros((classNum, dataNum))
    for i in range(classNum):
        tempResult[i] = theta_array[:, i].dot(x_array)
    classAns = []
    for i in range(dataNum):
        temp_maxIndex = 0
        tempMax = 0
        for j in range(classNum):
            if tempMax < tempResult[j, i]:
                temp_maxIndex = j
                tempMax = tempResult[j, i]
        classAns.append(temp_maxIndex)
    return np.max(tempResult, axis=0), np.array(classAns)




def print_Pic(theta_array, i, train_X, train_Y, dataTot):
    mk = ['*', 'o']
    cs = ['g', 'r']
    class_ans = H(train_X, theta_array, classNum)[1]
    loss_temp=Calculate_Loss(dataTot, train_X, train_Y, theta_array)
    print("============第 %d 轮===========" % (i))
    print("当前损失值为： %8.f" % (loss_temp))
    tempAccuracy = Calculate_Accuracy(class_ans, train_Y)
    print("当前的准确率为：%.8f" % (tempAccuracy))
    xL = 0
    xR = 1
    yL = 0
    yH = 1
    tempRange = 100
    meshX, meshY = np.meshgrid(np.linspace(xL, xR, tempRange), np.linspace(yL, yH, tempRange))
    tempAdd = np.ones((meshX.size,))
    meshData = np.vstack((tempAdd, meshX.flatten()))
    meshData = np.vstack((meshData, meshY.flatten()))
    meshPrdic = H(meshData, theta_array, classNum)[1]
    plt.contourf(meshX, meshY, meshPrdic.reshape(meshX.shape))
    plt.xlabel('x1')
    plt.ylabel('x2')
    for j in range(dataTot):
        plt.scatter(train_X[1, j], train_X[2, j],
                    marker=mk[int(train_Y[j])],
                    c=cs[int(train_Y[j])])
    plt.suptitle('Muti-Perceptron SGD')
    return tempAccuracy


def Multi_Perceptron_SGD(classNum, train_X, learning_rate, dataTot, train_Y):
    theta_array = np.zeros((train_X.shape[0], classNum))
    plt.ion()
    i = 0
    while 1:
        randomIndex = random.randint(0, dataTot - 1)
        for j in range(classNum):
            x = train_X[:, randomIndex]
            if int(H(x, theta_array, classNum)[1][0]) == j:
                error_H = 1
            else:
                error_H = 0
            if int(train_Y[randomIndex]) == j:
                error_Y = 1
            else:
                error_Y = 0
            ef = (error_H - error_Y) * x
            theta_array[:, j] -= learning_rate * ef
        temp_accuracy = print_Pic(theta_array, i, train_X, train_Y, dataTot)
        i += 1
        plt.pause(0.01)
        if temp_accuracy > 0.8:
            break
        plt.cla()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    dataTot, train_X, train_Y = load_data(fileName1, fileName2)
    train_X = data_process(train_X)
    classNum = 2
    Multi_Perceptron_SGD(classNum, train_X, 0.01, dataTot, train_Y)
