#############################
# author：919106840212 周运莲
# date：  2021-11-08
# note： 机器学习第一次课程作业
#############################
import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# 用的是 第二种 向量乘法   的形式

# 对数据集中的样本属性进行分割，制作X和Y矩阵
def feature_label_split(pd_data):

    row_cnt = pd_data.shape[0]  #行数
    column_cnt = len(pd_data.iloc[0, 0].split()) #列数
    # 生成新的X、Y矩阵
    X = np.empty([row_cnt, column_cnt - 1])  # 生成两个随机未初始化的矩阵
    Y = np.empty([row_cnt, 1])
    for i in range(0, row_cnt):
        row_array = pd_data.iloc[i, 0].split()
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y


# 把特征数据进行标准化为均匀分布
def uniform_norm(X_in):
    X_max = X_in.max(axis=0) #取每一列的最大值
    X_min = X_in.min(axis=0) #取每一列的最小值
    X = (X_in - X_min) / (X_max - X_min)
    return X, X_max, X_min


# 线性回归模型
class linear_regression():

    def fit(self, train_X_in, train_Y, learning_rate=0.01):
        # 样本个数、样本的属性个数
        case_cnt, feature_cnt = train_X_in.shape
        # X矩阵X0向量
        # np.c_按列连接矩阵
        # np.ones()创建一个case_cnt行的数组
        # 然后用np.c_ 连接

        train_X = np.c_[train_X_in, np.ones(case_cnt, )]
        # 初始化待调参数theta
        # np.zeros创建一个矩阵，arg0：行数；arg1：列数;初始值为 0
        # 这里就是 保存了 13+1个 theta参数，self (列向量)
        self.theta = np.zeros([feature_cnt + 1, 1])

        max_iter_num = sys.maxsize  # 最多迭代次数 ; sys.maxsize最大整数
        step = 0  # 当前已经迭代的次数
        pre_step = 0  # 上一次得到较好学习误差的迭代学习次数

        last_J_theta = sys.maxsize  # 上一次得到较好学习误差的误差函数值
        threshold_value = 1e-6  # 定义在得到较好学习误差之后截止学习的阈值
        stay_threshold_times = 10  # 定义 在得到较好学习误差之后 截止学习之前的学习次数

        # 迭代 repeat
        for step in range(0, max_iter_num):

            pred = train_X.dot(self.theta) # case_cnt * 1  (400*1)
            J_theta = sum((pred - train_Y) ** 2) / (2 * case_cnt)  # losses.append(J_theta)
            error_sum = np.empty([1,feature_cnt+1])    # print(error_sum.shape)
            # case_cnt次 向量乘法
            for index in range(0,case_cnt):
                error_sum += (pred[index]-train_Y[index])*((train_X[index]))
            # 更新
            self.theta -= learning_rate*error_sum.T/case_cnt
            # 检测损失函数的变化值，提前结束迭代
            if J_theta < last_J_theta - threshold_value:
                last_J_theta = J_theta
                pre_step = step
            elif step - pre_step > stay_threshold_times:
                final_step=step
                break

            # 定期打印，变化
            if step % 50 == 0:
                print("step %s: loss function value(J_theta) = %.5f" % (step, J_theta))
               # print(self.theta)

    def predict(self, X_in):
        case_cnt = X_in.shape[0]
        X = np.c_[X_in, np.ones(case_cnt, )]
        pred = X.dot(self.theta)
        return pred


# 主函数
if __name__ == "__main__":

    # 读取训练集和测试集文件
    train_data = pd.read_csv("housing-data/housing-data-train.csv", header=None)
    test_data = pd.read_csv("housing-data/housing-data-test.csv", header=None)
    losses=[]
    final_step=0

    # 对训练集和测试集进行X，Y分离
    train_X, train_Y = feature_label_split(train_data)
    test_X, test_Y = feature_label_split(test_data)

    # 对X（包括train_X, test_X）进行归一化处理，方便后续操作
    unif_trainX, X_max, X_min = uniform_norm(train_X)
    unif_testX = (test_X - X_min) / (X_max - X_min)
    # 开始计时
    start = time.perf_counter()
    #print("start time: "+start)
    # 模型训练
    model = linear_regression()
    model.fit(unif_trainX, train_Y, learning_rate=0.1)
    test_pred = model.predict(unif_testX)
    test_pred_J_theta = sum((test_pred - test_Y) ** 2) / (2 * unif_testX.shape[0])
    print("Test predict J_theta is %d" % (test_pred_J_theta))
    end = time.perf_counter()

    time_cost=end-start
    print("time-cost in method-2 (vector) is ")
    print(time_cost)
    # 画出损失函数的变化趋势
    # plot_x = np.arange(50,len(losses))
    # plot_y = np.array(losses)
    # plt.plot(plot_x, plot_y)
    # plt.show()

    # plt.subplot(1,3,2)
    plt.figure(figsize=(10, 10))
    plt.ylabel("Price")
    plt.plot(test_Y,color="blue",marker="o",label="true_price")
    plt.plot(test_pred,color="red",marker=".",label="predict")
    plt.legend()
    plt.show()
