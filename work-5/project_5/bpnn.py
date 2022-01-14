# 机器学习第五次作业：三层神经网络
# 919106840212 周运莲
# 2021-12-20

import numpy as np
from sklearn.model_selection import KFold

fileName1 = "./data/ex4x.dat"
fileName2 = "./data/ex4y.dat"


# 加载数据
def load_data(fileName1, fileName2):
    train_X = np.loadtxt(fileName1)
    train_Y = np.loadtxt(fileName2)
    return train_X, train_Y


# 把特征数据进行标准化为均匀分布
def uniform_norm(train_X):
    max_X = train_X.max(axis=0)  # 取每一列的最大值
    min_X = train_X.min(axis=0)  # 取每一列的最小值
    train_X = (train_X - min_X) / (max_X - min_X)
    return train_X


# 打乱数据
def _shuffle(train_X, train_Y):
    totalData = np.insert(train_X, 0, values=train_Y, axis=1)
    np.random.shuffle(totalData)
    train_X = totalData[:, 1:]
    train_Y = totalData[:, 0]
    return train_X, train_Y


input_size = 2
hidden_size = 3
output_size = 2



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, k):
    temp_sum = 0
    for i in range(2):
        temp_sum += np.exp(x[i])
    return np.exp(x[k]) / temp_sum


def H(w, b, x):
    return np.inner(w, x) + b


def BP_NN(train_X, train_Y, learning_rate,W,b):
    # 前馈计算每一层的净输入值 z和a 直到最后一层
    layer1 = np.ones(hidden_size)
    for i in range(hidden_size):
        layer1[i] = H(W[0][:, i], b[0][i], train_X)
    z1 = np.array(list(map(sigmoid, layer1)))

    layer2 = np.ones(output_size)
    for i in range(output_size):
        layer2[i] = H(W[1][:, i], b[1][i], z1)

    estimate_y0 = softmax(layer2, 0)
    estimate_y1 = softmax(layer2, 1)

    # 反向传播计算每一次的误差
    # 计算每一层的误差
    real_y0 = float(train_Y == 0)
    real_y1 = float(train_Y == 1)
    loss_value2 = np.array([estimate_y0 - real_y0, estimate_y1 - real_y1])
    # 更新参数
    temp_W1 = W[1]
    b[1] -= learning_rate * loss_value2
    diag = np.diag(z1 * (np.ones(hidden_size) - z1))
    z1 = np.mat([z1])
    loss_value2 = np.mat([loss_value2])
    W[1] -= learning_rate * np.dot(z1.T, loss_value2)
    loss_value1 = np.dot(np.dot(diag, temp_W1), loss_value2.T)
    # 更新参数
    W[0] -= learning_rate * np.dot(np.mat(train_X).T, loss_value1.T)
    b[0] -= learning_rate * loss_value1.T.A[0]
    if (estimate_y0 < estimate_y1):
        return 1
    else:
        return 0


def KFold_BP_ANN(train_X, train_Y, learning_rate=0.01):
    Precision = 0
    Recall = 0
    cnt_P = 0
    cnt_R = 0
    allData = np.insert(train_X, 0, values=train_Y, axis=1)  # x和y合并一下
    np.random.shuffle(allData)  # 打乱
    counter=0
    # k折 k= 5
    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 5 折
    for train_index, test_index in kf.split(allData):  # 将数据划分为5折
        trainSet = allData[train_index]  # train_index:选取的训练集数据下标,获取训练集
        testSet = allData[test_index]  # test_index: 选取的测试集数据下标 ,获取测试集
        # 训练集 4组*16条数据
        train_X = trainSet[:, 1:]
        train_Y = trainSet[:, 0]

        # 测试集  1组*16条数据
        test_X = testSet[:, 1:]
        test_Y = testSet[:, 0]

        # 初始化一些参数
        W = []
        b = []
        temp = [2, 3, 2]
        for i in range(2):
            w = np.ones((temp[i], temp[i + 1]))
            b.append(np.ones(temp[i + 1]))
            W.append(w)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # SGD 训练; 迭代5000次
        iteration = 5000
        for epoch in range(0, iteration):
            # 随机选取一个样本
            i = np.random.randint(0, 80 - 16)
            BP_NN(train_X[i], train_Y[i], learning_rate,W,b)
        counter+=1;
        print("============== 第 %d 次交叉验证结果 ================"%counter)
        print("W0的值：")
        print(W[0])
        print("W1的值")
        print(W[1])
        print("\n")
        # 预测剩下的那组数据
        for p in range(16):
            predict_Y = Predict(test_X[p],W,b)
            if (test_Y[p] == 0):
                if predict_Y == 0:
                    TN += 1
                else:
                    FN += 1
            else:
                if predict_Y == 1:
                    TP += 1
                else:
                    FP += 1
        if (TP + FP != 0):
            cnt_P += 1
            # print("第 %d 次的Precision = %.f"%(cnt_P,(TP / (TP + FP))))
            Precision += TP / (TP + FP)
        if (TP + FN != 0):
            cnt_R += 1
            # print("第 %d 次的Precision = %.f" % (cnt_R, (TP / (TP + FN))))
            Recall += TP / (TP + FN)
    Recall /= cnt_R
    Precision /= cnt_P

    print("采用5倍交叉最后得到的Precision：", Precision)
    print("采用5倍交叉最后得到的Recall：", Recall)
    print("采用5倍交叉最后得到的F1:", (Precision + Recall) / (2 * Precision * Recall))


# 预测
def Predict(test_X,W,b):
    a1 = np.ones(3)
    for i in range(3):
        a1[i] = H(W[0][:, i], b[0][i], test_X)
    z1 = np.array(list(map(sigmoid, a1)))
    a2 = np.ones(2)
    for i in range(2):
        a2[i] = H(W[1][:, i], b[1][i], z1)
    y0 = softmax(a2, 0)
    y1 = softmax(a2, 1)
    if (y0 < y1):
        return 1
    return 0


if __name__ == "__main__":
    train_X, train_Y = load_data(fileName1, fileName2)
    train_X = uniform_norm(train_X)
    KFold_BP_ANN(train_X, train_Y, learning_rate=0.01)
