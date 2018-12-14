import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dfx = pd.read_csv('linearX.csv')
dfy = pd.read_csv('linearY.csv')

x = dfx.values
y = dfy.values

X = (x - x.mean()) / x.std()
Y = y


def hypothesis(x, theta):
    return (theta[0] + theta[1] * x)


def error(X, Y, theta):
    m = X.shape[0]
    err = 0

    for i in range(m):
        hx = hypothesis(X[i], theta)
        err += (hx - Y[i])**2

    return err


def gradient(X, Y, theta):
    grad = np.zeros((2,))
    m = X.shape[0]
    for i in range(m):
        hx = hypothesis(X[i], theta)
        grad[0] += hx - Y[i]
        grad[1] += (hx - Y[i]) * X[i]

    return grad

#   Algorithm


def gradientDescent(X, Y, learning_rate=0.001):
    theta = np.zeros((2,))
    itr = 0
    max_itr = 100
    error_list = []
    theta_list = []

    while itr < max_itr:
        grad = gradient(X, Y, theta)
        e = error(X, Y, theta)
        error_list.append(e)
        theta_list.append((theta[0], theta[1]))
        theta[0] = theta[0] - learning_rate * grad[0]
        theta[1] = theta[1] - learning_rate * grad[1]
        itr += 1

    return theta, error_list, theta_list


final_theta, error_list, theta_list = gradientDescent(X, Y)
theta_list = np.array(theta_list)

X_Test = np.linspace(-2, 6, 10)

plt.scatter(X, Y)
plt.plot(X_Test, hypothesis(X_Test, final_theta), color='orange')
plt.show()

print("Loading Gradient Descent Algorithm Chart")

T0 = np.arange(-2, 3, 0.01)
T1 = T0
T0, T1 = np.meshgrid(T0, T1)

J = np.zeros(T0.shape)

m = T0.shape[0]
n = T1.shape[0]

for i in range(m):
    for j in range(n):
        J[i, j] = np.sum((Y - T1[i, j] * X - T0[i, j])**2)

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.scatter(theta_list[:, 0], theta_list[:, 1], error_list, c='k')
axes.plot_surface(T0, T1, J, cmap='rainbow', alpha=0.5)
plt.show()
