# 机器学习

# 1.机器学习算法

# 1）PCA算法 主成分分析

#!/usr/bin/env python
# encoding=gbk
# coding=utf-8

import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)#相乘


# 调用
pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)  # 输出降维后的数据


print('------------------------------------')

# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):
        '''
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


# if __name__ == '__main__':
'10样本3特征的样本集, 行为样例，列为特征维度'
X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
K = np.shape(X)[1] - 1
print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
pca = CPCA(X, K)

print('------------------------------------')


# 2）k-means算法
# coding=utf-8
from sklearn.cluster import KMeans

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
     ]

# 输出数据集
# print(X)

"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

# 输出完整Kmeans函数，包括很多省略参数
print(clf)
# 输出聚类预测结果
print("y_pred = ", y_pred)

"""
第三部分：可视化绘图
"""

import numpy as np
import matplotlib.pyplot as plt

# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()

# cluster.py层次聚类
# 导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()

# 密度聚类
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN


iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X.shape)
# 绘制数据分布图

plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  


dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

print('------------------------------------')

# 3)最小二乘
import pandas as pd

sales=pd.read_csv('data/train_data.csv',sep='\s*,\s*',engine='python')  #读取CSV
X=sales['X'].values    #存csv的第一列
Y=sales['Y'].values    #存csv的第二列

#初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4       ####你需要根据的数据量进行修改

#循环累加
for i in range(n):
    s1 = s1 + X[i]*Y[i]     #X*Y，求和
    s2 = s2 + X[i]          #X的和
    s3 = s3 + Y[i]          #Y的和
    s4 = s4 + X[i]*X[i]     #X**2，求和

#计算斜率和截距
k = (s2*s3-n*s1)/(s2*s2-s4*n)
b = (s3 - k*s2)/n
print("Coeff: {} Intercept: {}".format(k, b))
#y=1.4x+3.5

# 4）SVM算法
# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

class1 = np.array([[1, 1], [1, 3], [2, 1], [1, 2], [2, 2]])
class2 = np.array([[4, 4], [5, 5], [5, 4], [5, 3], [4, 5], [6, 4]])

plt.figure(figsize=(6, 4), dpi=120)

plt.title('Decision Boundary')

plt.xlim(0, 8)
plt.ylim(0, 6)
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.scatter(class1[:, 0], class1[:, 1], marker='o')
plt.scatter(class2[:, 0], class2[:, 1], marker='s')
plt.plot([1, 5], [5, 1], '-r')
plt.arrow(4, 4, -1, -1, shape='full', color='r')
plt.plot([3, 3], [0.5, 6], '--b')
plt.arrow(4, 4, -1, 0, shape='full', color='b', linestyle='--')
plt.annotate(r'margin 1',
             xy=(3.5, 4), xycoords='data',
             xytext=(3.1, 4.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'margin 2',
             xy=(3.5, 3.5), xycoords='data',
             xytext=(4, 3.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'support vector',
             xy=(4, 4), xycoords='data',
             xytext=(5, 4.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'support vector',
             xy=(2, 2), xycoords='data',
             xytext=(0.5, 1.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.savefig("result/SVM1.jpg")

plt.show()

# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

class1 = np.array([[1, 1], [1, 3], [2, 1], [1, 2], [2, 2]])
class2 = np.array([[4, 4], [5, 5], [5, 4], [5, 3], [4, 5], [6, 4]])

plt.figure(figsize=(6, 4), dpi=120)

plt.title('Support Vector Machine')

plt.xlim(0, 8)
plt.ylim(0, 6)
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.scatter(class1[:, 0], class1[:, 1], marker='o')
plt.scatter(class2[:, 0], class2[:, 1], marker='s')
plt.plot([1, 5], [5, 1], '-r')
plt.plot([0, 4], [4, 0], '--b', [2, 6], [6, 2], '--b')
plt.arrow(4, 4, -1, -1, shape='full', color='b')
plt.annotate(r'$w^T x + b = 0$',
             xy=(5, 1), xycoords='data',
             xytext=(6, 1), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'$w^T x + b = 1$',
             xy=(6, 2), xycoords='data',
             xytext=(7, 2), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'$w^T x + b = -1$',
             xy=(3.5, 0.5), xycoords='data',
             xytext=(4.5, 0.2), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'd',
             xy=(3.5, 3.5), xycoords='data',
             xytext=(2, 4.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'A',
             xy=(4, 4), xycoords='data',
             xytext=(5, 4.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.savefig("result/SVM2.jpg")

plt.show()

# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs

plt.figure(figsize=(10, 4), dpi=140)

# sub plot 1
plt.subplot(1, 2, 1)

X, y = make_blobs(n_samples=100,
                  n_features=2,
                  centers=[(1, 1), (2, 2)],
                  random_state=4,
                  shuffle=False,
                  cluster_std=0.4)

plt.title('Non-linear Separatable')

plt.xlim(0, 3)
plt.ylim(0, 3)
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='o')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='s')
plt.plot([0.5, 2.5], [2.5, 0.5], '-r')

# sub plot 2
plt.subplot(1, 2, 2)

class1 = np.array([[1, 1], [1, 3], [2, 1], [1, 2], [2, 2], [1.5, 1.5], [1.2, 1.7]])
class2 = np.array([[4, 4], [5, 5], [5, 4], [5, 3], [4, 5], [6, 4], [5.5, 3.5], [4.5, 4.5], [2, 1.5]])

plt.title('Slack Variable')

plt.xlim(0, 7)
plt.ylim(0, 7)
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.scatter(class1[:, 0], class1[:, 1], marker='o')
plt.scatter(class2[:, 0], class2[:, 1], marker='s')
plt.plot([1, 5], [5, 1], '-r')
plt.plot([0, 4], [4, 0], '--b', [2, 6], [6, 2], '--b')
plt.arrow(2, 1.5, 2.25, 2.25, shape='full', color='b')
plt.annotate(r'violate margin rule.',
             xy=(2, 1.5), xycoords='data',
             xytext=(0.2, 0.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'normal sample. $\epsilon = 0$',
             xy=(4, 5), xycoords='data',
             xytext=(4.5, 5.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'$\epsilon > 0$',
             xy=(3, 2.5), xycoords='data',
             xytext=(3, 1.5), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.savefig("result/SVM3.jpg")

plt.show()

# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 4), dpi=120)

plt.title('Cost')

plt.xlim(0, 4)
plt.ylim(0, 2)
plt.xlabel('$y^{(i)} (w^T x^{(i)} + b)$')
plt.ylabel('Cost')
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.plot([0, 1], [1.5, 0], '-r')
plt.plot([1, 3], [0.015, 0.015], '-r')
plt.annotate(r'$J_i = R \epsilon_i$ for $y^{(i)} (w^T x^{(i)} + b) \geq 1 - \epsilon_i$',
             xy=(0.7, 0.5), xycoords='data',
             xytext=(1, 1), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'$J_i = 0$ for $y^{(i)} (w^T x^{(i)} + b) \geq 1$',
             xy=(1.5, 0), xycoords='data',
             xytext=(1.8, 0.2), fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.savefig("result/SVM4.jpg")

plt.show()

# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 4), dpi=144)

class1 = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [3, 2], [4, 1], [5, 1]])
class2 = np.array([[2.2, 4], [1.5, 5], [1.8, 4.6], [2.4, 5], [3.2, 5], [3.7, 4], [4.5, 4.5], [5.4, 3]])

# sub plot 1
plt.subplot(1, 2, 1)

plt.title('Non-linear Separatable in Low Dimension')

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.yticks(())
plt.xlabel('X1')
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')

plt.scatter(class1[:, 0], np.zeros(class1[:, 0].shape[0]) + 0.05, marker='o')
plt.scatter(class2[:, 0], np.zeros(class2[:, 0].shape[0]) + 0.05, marker='s')

# sub plot 2
plt.subplot(1, 2, 2)

plt.title('Linear Separatable in High Dimension')

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.xlabel('X1')
plt.ylabel('X2')
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.scatter(class1[:, 0], class1[:, 1], marker='o')
plt.scatter(class2[:, 0], class2[:, 1], marker='s')
plt.plot([1, 5], [3.8, 2], '-r')

plt.savefig("result/SVM5.jpg")

plt.show()

# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel(x, mean, sigma):
    return np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))


x = np.linspace(0, 6, 500)
mean = 1
sigma1 = 0.1
sigma2 = 0.3

plt.figure(figsize=(10, 3), dpi=144)

# sub plot 1
plt.subplot(1, 2, 1)
plt.title('Gaussian for $\sigma={0}$'.format(sigma1))

plt.xlim(0, 2)
plt.ylim(0, 1.1)
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.plot(x, gaussian_kernel(x, mean, sigma1), 'r-')

# sub plot 2
plt.subplot(1, 2, 2)
plt.title('Gaussian for $\sigma={0}$'.format(sigma2))

plt.xlim(0, 2)
plt.ylim(0, 1.1)
ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
ax.spines['right'].set_color('none')  # 隐藏坐标轴
ax.spines['top'].set_color('none')

plt.plot(x, gaussian_kernel(x, mean, sigma2), 'r-')

plt.savefig("result/SVM6.jpg")

plt.show()



