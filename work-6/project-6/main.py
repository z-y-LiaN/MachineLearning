# 919106840212-周运莲-机器学习第6次作业
# 基于K-means的数据聚类
# 2021-12-28
import matplotlib.pyplot as plt
import numpy as np
import random

train_data = np.loadtxt('ex4x.dat')

minX = train_data[:, 0].min()
minY = train_data[:, 1].min()
maxX = train_data[:, 0].max()
maxY = train_data[:, 1].max()

centers = [[random.uniform(minX, maxX), random.uniform(minY, maxY)],[random.uniform(minX, maxX), random.uniform(minY, maxY)]]

cnt = 0
while(cnt <= 20):
    cnt += 1
    sumX_sumY = np.zeros((2, 2))
    plt.scatter(centers[0][0], centers[0][1], color="r", marker='x')
    plt.scatter(centers[1][0], centers[1][1], color="b", marker='x')
    cnt0 = 0
    cnt1 = 1
    for i in range(80):
        dist0 = np.linalg.norm(train_data[i] - centers[0])
        dist1 = np.linalg.norm(train_data[i] - centers[1])
        if dist0 < dist1:
            cnt0 += 1
            sumX_sumY[0] += train_data[i]
            plt.scatter(train_data[i][0], train_data[i][1], color='r')
        else:
            cnt1 += 1
            sumX_sumY[1] += train_data[i]
            plt.scatter(train_data[i][0], train_data[i][1], color='b')
    centers[0] = sumX_sumY[0] / cnt0
    centers[1] = sumX_sumY[1] / cnt1
    print("第 %d 次："%cnt)
    print(centers)
    # plt.pause(15)
    plt.pause(0.01)
    plt.cla()
plt.ioff()
plt.show()
print("最终结果为：")
print(centers)