################################
# 919106840212 周运莲
# 机器学习第四次作业
# SVM 采用线性核函数，进行k折交叉验证
# 2021-12-15
################################
from sklearn import svm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

fileName1 = "./data/ex4x.dat"
fileName2 = "./data/ex4y.dat"


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
    # # 插入x0=1这一列
    # train_X = np.insert(train_X, 0, values=1.0, axis=1)
    print(train_X)
    return train_X


if __name__ == "__main__":
    """
        数据处理
    """
    train_X, train_Y = load_data(fileName1, fileName2)
    train_X = uniform_norm(train_X)
    """
        SVM
    """
    clf = svm.SVC(kernel='linear')  # 线性核函数
    clf.fit(train_X, train_Y)  # 拟合
    print(clf.support_vectors_)  # 打印支持向量
    # print(clf.support_)
    print("每类支持向量的个数")
    print(clf.n_support_)  # 每类的支持向量个数
    """
        作图
    """
    w = clf.coef_[0]  # 获取w
    a = -w[0] / w[1]  # 斜率：a
    x = np.linspace(0, 1, 50)
    y_mid = a * x - (clf.intercept_[0]) / w[1]  # 求出过切线的点：clf.intercept_[0]

    b = clf.support_vectors_[0]  # 第一个支持向量
    y_down = a * x + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]  # 最后一个支持向量
    y_up = a * x + (b[1] - a * b[0])
    print("w: ", w)
    print("a: ", a)
    pl.plot(x, y_mid, 'k-')
    pl.plot(x, y_down, 'k--')
    pl.plot(x, y_up, 'k--')
    train_X = train_X.T
    dataTot = 80
    plt.scatter(train_X[0, :40], train_X[1, :40], color='blue')
    plt.scatter(train_X[0, 40:], train_X[1, 40:], color='red')
    plt.title('SVM——Kernel function')
    plt.show()
    """
        80个数据，分成8组，每组10个，进行交叉验证
    """
    # cross_val_score自动数据进行k折交叉验证，
    # 一组10个，cv=8,即表示8折交叉验证，
    train_X = train_X.T
    scores = cross_val_score(clf, train_X, train_Y, cv=8)  #
    # 输出8个指标值
    print("每次进行交叉验证得分:{}".format(scores))
    print("平均得分:{:2f}".format(scores.mean()))

#     cv_scores = []  # 用来放每个模型的结果值
#     k_range = range(1, 8)
#     for n in k_range:
#         scores = cross_val_score(clf, train_X, train_Y, cv=8)  # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值，具体使用参考下面。
#         cv_scores.append(scores)
# plt.plot(k_range, cv_scores)
# plt.xlabel('K')
# plt.ylabel('Accuracy')  # 通过图像选择最好的参数
# plt.show()
