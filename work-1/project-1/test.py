import numpy as np

import numpy as np

fileName="housing.data"
data = np.loadtxt(fileName)
m=400
n=13
x_train=data[:m,:n]
y_train=data[:m,n:]
x_test=data[m:,:n]
y_test=data[m:,n:]
x0_train=np.ones(m)
x_train=np.insert(x_train,0,x0_train,axis=1)
print(x_train[1])
print(x_train[1][13])
m_test=x_test.shape[0]
x0_test=np.ones(m_test)
x_test=np.insert(x_test,0,x0_test,axis=1)

alpha=1.0e-6
epoch_num=10000

theta_array=np.random.randn(n+1)
for epoch in range(epoch_num):
    gradient_sum_array =np.zeros(n+1)
    for i in range(m):
        h_theta =np.dot(theta_array,x_train[i])
        gradient =(h_theta-y_train[i])*x_train[i]
        gradient_sum_array+=gradient
    theta_array=theta_array-alpha/m*gradient_sum_array
    print(theta_array)

     #compute cost value
    J_theta=0.0
    for i in range(m):
        J_theta = np.dot(theta_array,x_train[i])
        J_theta += (h_theta-y_train[i])**2
    J_theta=J_theta/m
    mse=0.0
    for i in range(m_test):
        h_theta=np.dot(theta_array,x_test[i])
        mse+=(h_theta-y_test[i])**2
    mse=mse/m


