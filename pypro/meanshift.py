# 设想在一个有N个样本点的特征空间初始确定一个中心点center,
# 计算在设置的半径为D的圆形空间内所有的点（xi）与中心点center的向量,计算整个圆形空间内所有向量的平均值，
# 得到一个偏移均值将中心点center移动到偏移均值位置重复移动,直到满足一定条件结束

import matplotlib.pyplot as plt
import math

# 高斯核函数
def cal_Gaussian(x, h=1):
    molecule = x * x
    denominator = 2 * h * h
    left = 1 / (math.sqrt(2 * math.pi) * h)
    return left * math.exp(-molecule / denominator)

x = []

for i in range(-40,40):
    x.append(i * 0.5);

score_1 = []
score_2 = []
score_3 = []
score_4 = []

for i in x:
    score_1.append(cal_Gaussian(i,1))
    score_2.append(cal_Gaussian(i,2))
    score_3.append(cal_Gaussian(i,3))
    score_4.append(cal_Gaussian(i,4))

plt.plot(x, score_1, 'b--', label="h=1")
plt.plot(x, score_2, 'k--', label="h=2")
plt.plot(x, score_3, 'g--', label="h=3")
plt.plot(x, score_4, 'r--', label="h=4")

plt.legend(loc="upper right")
plt.xlabel("x")
plt.ylabel("N")
plt.show()


import numpy as np

# Input data set
X = np.array([
    [-4, -3.5], [-3.5, -5], [-2.7, -4.5],
    [-2, -4.5], [-2.9, -2.9], [-0.4, -4.5],
    [-1.4, -2.5], [-1.6, -2], [-1.5, -1.3],
    [-0.5, -2.1], [-0.6, -1], [0, -1.6],
    [-2.8, -1], [-2.4, -0.6], [-3.5, 0],
    [-0.2, 4], [0.9, 1.8], [1, 2.2],
    [1.1, 2.8], [1.1, 3.4], [1, 4.5],
    [1.8, 0.3], [2.2, 1.3], [2.9, 0],
    [2.7, 1.2], [3, 3], [3.4, 2.8],
    [3, 5], [5.4, 1.2], [6.3, 2]
])


def mean_shift(data, radius=2.0):
    clusters = []
    for i in range(len(data)):
        cluster_centroid = data[i]
        cluster_frequency = np.zeros(len(data))

        # Search points in circle
        while True:
            temp_data = []
            for j in range(len(data)):
                v = data[j]
                # Handle points in the circles
                if np.linalg.norm(v - cluster_centroid) <= radius:
                    temp_data.append(v)
                    cluster_frequency[i] += 1

            # Update centroid
            old_centroid = cluster_centroid
            new_centroid = np.average(temp_data, axis=0)
            cluster_centroid = new_centroid
            # Find the mode
            if np.array_equal(new_centroid, old_centroid):
                break
        # Combined 'same' clusters
        has_same_cluster = False
        for cluster in clusters:
            if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                has_same_cluster = True
                cluster['frequency'] = cluster['frequency'] + cluster_frequency
                break
        if not has_same_cluster:
            clusters.append({
                'centroid': cluster_centroid,
                'frequency': cluster_frequency
            })
    print('clusters (', len(clusters), '): ', clusters)
    clustering(data, clusters)
    show_clusters(clusters, radius)


# Clustering data using frequency
def clustering(data, clusters):
    t = []
    for cluster in clusters:
        cluster['data'] = []
        t.append(cluster['frequency'])
    t = np.array(t)
    # Clustering
    for i in range(len(data)):
        column_frequency = t[:, i]
        cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
        clusters[cluster_index]['data'].append(data[i])


# Plot clusters
def show_clusters(clusters, radius):
    colors = 10 * ['r', 'g', 'b', 'k', 'y']
    plt.figure(figsize=(5, 5))
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    plt.scatter(X[:, 0], X[:, 1], s=20)
    theta = np.linspace(0, 2 * np.pi, 800)
    for i in range(len(clusters)):
        cluster = clusters[i]
        data = np.array(cluster['data'])
        plt.scatter(data[:, 0], data[:, 1], color=colors[i], s=20)
        centroid = cluster['centroid']
        plt.scatter(centroid[0], centroid[1], color=colors[i], marker='x', s=30)
        x, y = np.cos(theta) * radius + centroid[0], np.sin(theta) * radius + centroid[1]
        plt.plot(x, y, linewidth=1, color=colors[i])
    plt.show()


mean_shift(X, 2.5)


print('**************************')
import numpy as np
import math

MIN_DISTANCE = 0.00001  # 最小误差
def euclidean_dist(pointA, pointB):
    # 计算pointA和pointB之间的欧式距离
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)
def gaussian_kernel(distance, bandwidth):
    ''' 高斯核函数
    :param distance: 欧氏距离计算函数
    :param bandwidth: 核函数的带宽
    :return: 高斯函数值
    '''
    m = np.shape(distance)[0]  # 样本个数
    right = np.mat(np.zeros((m, 1)))#矩阵
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))
    gaussian_val = left * right
    return gaussian_val

def shift_point(point, points, kernel_bandwidth):
    '''
    计算均值漂移点
    :param point: 需要计算的点
    :param points: 所有的样本点
    :param kernel_bandwidth: 核函数的带宽
    :return:
        point_shifted：漂移后的点
    '''
    points = np.mat(points)
    m = np.shape(points)[0]  # 样本个数
    # 计算距离
    point_distances = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_distances[i, 0] = euclidean_dist(point, points[i])
    # 计算高斯核
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)

    # 计算分母
    all = 0.0
    for i in range(m):
        all += point_weights[i, 0]
    # 均值偏移
    point_shifted = point_weights.T * points / all
    return point_shifted


def group_points(mean_shift_points):
    '''
    计算所属的类别
    :param mean_shift_points:漂移向量
    :return: group_assignment：所属类别
    '''
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))
        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])
    return group_assignment

def train_mean_shift(points, kernel_bandwidth=2):
    '''
    训练Mean Shift模型
    :param points: 特征数据
    :param kernel_bandwidth: 核函数带宽
    :return:
        points：特征点
        mean_shift_points：均值漂移点
        group：类别
    '''
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0
    m = np.shape(mean_shift_points)[0]  # 样本的个数
    need_shift = [True] * m  # 标记是否需要漂移
    # 计算均值漂移向量
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        print("iteration : " + str(iteration))
        for i in range(0, m):
            # 判断每一个样本点是否需要计算偏置均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)  # 对样本点进行偏移
            dist = euclidean_dist(p_new, p_new_start)
             # 计算该点与漂移后的点之间的距离
            if dist > max_min_dist:  # 记录是有点的最大距离
                max_min_dist = dist
            if dist < MIN_DISTANCE:  # 不需要移动
                need_shift[i] = False
            mean_shift_points[i] = p_new
    # 计算最终的group
    group = group_points(mean_shift_points)
    # 计算所属的类别
    return np.mat(points), mean_shift_points, group


import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

data = []
f = open("D:\HYD\code\pypro\data/iris_training.csv", 'r')#换成自己的数据所在的路径
for line in f:
    data.append([float(line.split(',')[0]), float(line.split(',')[1])])
data = np.array(data)

# 通过下列代码可自动检测bandwidth值
# 从data中随机选取1000个样本，计算每一对样本的距离，然后选取这些距离的0.2分位数作为返回值，当n_samples很大时，这个函数的计算量是很大的。

bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=1000)
print(bandwidth)

# bin_seeding设置为True就不会把所有的点初始化为核心位置，从而加速算法

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# 计算类别个数
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters)

# 画图
import matplotlib.pyplot as plt
from itertools import cycle
plt.figure(1)
plt.clf()  # 清楚上面的旧图形
# cycle把一个序列无限重复下去
colors = cycle('bgrcmyk')
for k, color in zip(range(n_clusters), colors):
    # current_member表示标签为k的记为true 反之false

    current_member = labels == k
    cluster_center = cluster_centers[k]

    # 画点
    plt.plot(data[current_member, 0], data[current_member, 1], color + '.')

    #画圈
    plt.plot(cluster_center[0], cluster_center[1], 'o',
             markerfacecolor=color,  #圈内颜色
             markeredgecolor='k',  #圈边颜色
             markersize=14)  #圈大小
plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()
