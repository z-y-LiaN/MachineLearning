import sys
import numpy as np
import matplotlib.pyplot as plt

fileName1 = "./dataset/ex4x.dat"
fileName2 = "./dataset/ex4y.dat"


# sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 加载数据
def load_data(fileName1, fileName2):
    train_X = np.loadtxt(fileName1)
    train_Y = np.loadtxt(fileName2)
    return train_X, train_Y


# 把特征数据进行标准化为均匀分布,
def uniform_norm(train_X):
    max_X = train_X.max(axis=0)  # 取每一列的最大值
    min_X = train_X.min(axis=0)  # 取每一列的最小值
    train_X = (train_X - min_X) / (max_X - min_X)
    # 插入x0=1这一列
    train_X = np.insert(train_X, 0, values=1.0, axis=1)
    print(train_X)
    return train_X


# 损失函数
def loss(train_X, train_Y, theta_array):
    H = sigmoid(train_X.dot(theta_array))
    # J = -1.0*(1.0/80)* (np.log(H).T.dot(train_Y) + np.log(1-H).T.dot(1-train_Y))
    J = -1.0 * (1.0 / 80) * (train_Y.T.dot(np.log(H)) + (1 - train_Y).T.dot(np.log(1 - H)))
    return J


def logistic_GD(train_X, train_Y, learning_rate, iteration):
    # s_c = 80, f_c = 3 ( x0=1,x1,x2 )
    sample_cnt, feature_cnt = train_X.shape
    theta_array = np.zeros(3)
    # 矩阵形式
    # for i in range(0, iteration):
    #     # 矩阵的形式
    #     H_theta = sigmoid(np.dot(theta.T, train_X))  # 1*80
    #     # print(H_theta)
    #     gradient = np.dot(train_X, (H_theta - train_Y).T)  # 3*80 80*1
    #     theta = theta - learning_rate * gradient / 80
    #
    # 向量形式
    # 如果指定loss与preloss之差
    preloss_value = 0.0
    max_iteration = sys.maxsize
    for step in range(max_iteration):
        gradient_sum = np.zeros(3)
        for i in range(sample_cnt):
            H_i = sigmoid(np.dot(theta_array, train_X[i]))
            gradient = (H_i - train_Y[i]) * train_X[i] / sample_cnt
            gradient_sum += gradient
        theta_array -= learning_rate * gradient_sum
        loss_value = loss(train_X, train_Y, theta_array)
        print("logistic GD: 第 %d 次 迭代 loss = %.8f "%(step,loss_value))
        # print(loss_value)
        if (abs(preloss_value - loss_value) < 1e-6):
            break
        else:
            preloss_value = loss_value
    # 如果指定迭代次数
    # for step in range(iteration):
    #     gradient_sum = np.zeros(3)
    #     for i in range(sample_cnt):
    #         H_i = sigmoid(np.dot(theta_array, train_X[i]))
    #         gradient = (H_i - train_Y[i]) * train_X[i] / sample_cnt
    #         gradient_sum += gradient
    #     theta_array -= learning_rate * gradient_sum
    #     print("loss = ")
    #     print(loss(train_X,train_Y,theta_array))
    return theta_array


def logistic_SGD(train_X, train_Y, learning_rate, iteration):
    # s_c = 80, f_c = 3 ( x0=1,x1,x2 )
    sample_cnt, feature_cnt = train_X.shape
    theta_array = np.zeros(3)
    # 指定loss-preloss的方式
    preloss_value = 0.0
    max_iteration = sys.maxsize
    for step in range(max_iteration):
        for i in range(sample_cnt):
            H_i = sigmoid(np.dot(theta_array, train_X[i]))
            for j in range(0, feature_cnt):
                theta_array[j] -= learning_rate * (H_i - train_Y[i]) * train_X[i][j]
        loss_value = loss(train_X, train_Y, theta_array)
        print("logistic SGD: 第 %d 次 迭代 loss = %.8f " % (step, loss_value))
        # print(loss_value)
        if (abs (preloss_value - loss_value) < 1e-6):
            break
        else:
            preloss_value = loss_value
    # 指定迭代次数的方式
    # for step in range(iteration):
    #     for i in range(sample_cnt):
    #         H_i = sigmoid(np.dot(theta_array, train_X[i]))
    #         for j in range(0, feature_cnt):
    #             theta_array[j] -= learning_rate * (H_i - train_Y[i]) * train_X[i][j]
    return theta_array


def print_img(train_X, theta_array0, theta_array1):
    plt.cla()
    train_X = train_X.T
    plt.scatter(train_X[1, :40], train_X[2, :40], color='blue')
    plt.scatter(train_X[1, 40:], train_X[2, 40:], color='red')
    x1 = np.array([0, 1])
    # print(theta_array[0])
    # print(theta_array[1])
    # print(theta_array[2])
    plt.plot(x1, -((theta_array0[1] / theta_array0[2]) * x1 + theta_array0[0] / theta_array0[2]))
    plt.plot(x1, -((theta_array1[1] / theta_array1[2]) * x1 + theta_array1[0] / theta_array1[2]))
    plt.show()


if __name__ == "__main__":
    train_X, train_Y = load_data(fileName1, fileName2)
    train_X = uniform_norm(train_X)
    print("======================logistic GD begin=======================")
    theta_array0 = logistic_GD(train_X, train_Y, 1e-2, 10000)
    print("======================logistic GD end=======================")
    print("\n")
    print("======================logistic SGD begin=======================")
    theta_array1 = logistic_SGD(train_X, train_Y, 1e-2, 10000)
    print("======================logistic SGD end=======================")
    print_img(train_X, theta_array0, theta_array1)
