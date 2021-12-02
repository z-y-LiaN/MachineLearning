import sys
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

fileName1 = "./dataset/ex4x.dat"
fileName2 = "./dataset/ex4y.dat"

def softmax(x):      # x: 列
     # print("x.shape = ")
     # print(x.shape)
     # print(x.type())
     t= np.exp(x)/np.sum(np.exp(x),axis=1)
     # print("t.type = ")
     # print(t.type())
     return t

# 加载数据
def load_data(fileName1,fileName2):
    train_X = np.loadtxt(fileName1)
    train_Y = np.loadtxt(fileName2, dtype=int)
    return train_X, train_Y

# 数据处理,
def uniform_norm(train_X,train_Y):
    max_X = train_X.max(axis=0) # 取每一列的最大值
    min_X = train_X.min(axis=0) # 取每一列的最小值
    train_X = (train_X - min_X) / (max_X - min_X)
    #插入x0=1这一列
    train_X = np.insert(train_X, 0, values=1.0, axis=1)
    # train_X = np.mat(train_X)
    #把train_Y矩阵 处理成onehot,  80*2
    train_Y_onehot=np.zeros((train_Y.shape[0],2))
    eyes_mat = np.eye(2)
    # 按分类数2生成对角线为2的单位阵
    for i in range(0,train_Y.shape[0]):
        train_Y_onehot[i]=eyes_mat[train_Y[i]]
    # train_Y_onehot = np.mat(train_Y_onehot)
    # print(train_X)
    return train_X,train_Y_onehot

# def loss(train_X,train_Y_onehot,theta_mat):
#     loss_value = - (1 / 80) * np.sum(train_Y_onehot * np.log(softmax(train_X*theta_mat).T))
#     return loss_value

def softmax_GD(train_X,train_Y_onehot,learning_rate,iteration):
    # s_c = 80, f_c = 3 ( x0=1,x1,x2 )
    sample_cnt,feature_cnt=shape(train_X)
    train_X =  np.mat(train_X)
    train_Y_onehot=np.mat(train_Y_onehot)
    # train_Y=np.mat(train_Y)
    # theta_mat=zeros((feature_cnt,2)) #3*2
    theta_mat=np.random.randn(feature_cnt,2)
    preloss_value = 0.0
    max_iteration = sys.maxsize
    # for epoch in range (max_iteration):
    #     H = softmax(train_X * theta_mat)
    #     error = H - train_Y_onehot  # 80*2-80*2
    #     theta_mat = theta_mat - learning_rate * (train_X.T) * error  # 3*80 80*2
    #     loss_value=loss(train_X, train_Y_onehot, theta_mat)
    #     print("softmax GD: 第 %d 次 迭代 loss = %.10f " % (epoch, loss_value))
    #     if (abs(preloss_value - loss_value) < 1e-6):
    #         break
    #     else:
    #         preloss_value = loss_value
    for step in range(0,iteration):
        # H=softmax(np.dot(train_X,theta_mat)) # 80*3,3*2 = 80*2
        H =softmax(train_X*theta_mat)
        error=H-train_Y_onehot #80*2-80*2
        # print(error.type())
        # print((train_X.T*error).type())
        theta_mat =theta_mat-learning_rate*(train_X.T)*error # 3*80 80*2
        # print(theta_mat.type())
        # print(loss(train_X,train_Y_onehot,theta_mat))
    return theta_mat.getA()

def softmax_SGD(train_X,train_Y_onehot,learning_rate,iteration):
    # s_c = 80, f_c = 3 ( x0=1,x1,x2 )
    sample_cnt, feature_cnt = shape(train_X)
    train_X = np.mat(train_X)
    train_Y_onehot = np.mat(train_Y_onehot)
    # train_Y=np.mat(train_Y)
    # theta_mat=zeros((feature_cnt,2)) #3*2
    theta_mat = np.random.randn(feature_cnt, 2)
    for step in range(0, iteration):
        # rand = random.randint(0,1)
        # rand=0
        # for i in range(0,sample_cnt):
            i=np.random.randint(0,sample_cnt)
            H=softmax(train_X[i]*theta_mat)
            # print(H) #1*2
            error= H[0]-train_Y_onehot[i]
            theta_mat=theta_mat-learning_rate*train_X[i].T*error[0]
    return theta_mat.getA()

def print_img(train_X, theta_mat_GD,theta_mat_SGD):
    plt.cla()
    train_X = train_X.T
    plt.scatter(train_X[1, :40], train_X[2, :40], color='blue')
    plt.scatter(train_X[1, 40:], train_X[2, 40:], color='red')
    x1 = np.array([0, 1])
    # print(theta_array[0])
    # print(theta_array[1])
    # print(theta_array[2])
    para0= theta_mat_GD[0, 0] - theta_mat_GD[0, 1]
    para1= theta_mat_GD[1, 0] - theta_mat_GD[1, 1]
    para2= theta_mat_GD[2, 0] - theta_mat_GD[2, 1]
    _para0 = theta_mat_SGD[0, 0] - theta_mat_SGD[0, 1]
    _para1 = theta_mat_SGD[1, 0] - theta_mat_SGD[1, 1]
    _para2 = theta_mat_SGD[2, 0] - theta_mat_SGD[2, 1]
    plt.plot(x1, -((para1/ para2) * x1 + para0 / para2))
    plt.plot(x1, -((_para1 / _para2) * x1 + _para0 / _para2))
    plt.show()


if __name__ == "__main__":
    # fig = plt.figure()
    train_X,train_Y =load_data(fileName1,fileName2)
    train_X,train_Y_onehot= uniform_norm(train_X,train_Y)
    theta_GD=softmax_GD(train_X,train_Y_onehot,1e-2,10000)
    print(theta_GD)
    theta_SGD=softmax_SGD(train_X,train_Y_onehot,1e-2,10000)
    print(theta_SGD)
    print_img(train_X,theta_GD,theta_SGD)




