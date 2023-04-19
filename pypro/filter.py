# 滤波算法

# 1.噪声
# 1）椒盐噪声
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt

def addsalt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声

    return img_


img = cv2.imread('data/net.png')

SNR_list = [0.9, 0.7, 0.5, 0.3]
sub_plot = [221, 222, 223, 224]

plt.figure(1)
for i in range(len(SNR_list)):
    plt.subplot(sub_plot[i])
    img_s = addsalt_pepper(img.transpose(2, 1, 0), SNR_list[i])     # c,
    img_s = img_s.transpose(2, 1, 0)
    cv2.imshow('PepperandSalt', img_s)
    cv2.waitKey(0)
    plt.imshow(img_s[:,:,::-1])     # bgr --> rgb
    plt.title('add salt pepper noise(SNR={})'.format(SNR_list[i]))

plt.show()

# 2）高斯噪声
import numpy as np
import pylab as pl


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


t = np.arange(0, 1000000) * 0.1
x = np.sin(t)
n = wgn(x, 6)
xn = x + n  # 增加了6dBz信噪比噪声的信号
pl.subplot(211)
pl.hist(n, bins=100, density=True)
pl.subplot(212)
pl.psd(n)
pl.show()



import random
import cv2

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
# Read image
img = cv2.imread("data/net.png")
# 添加椒盐噪声，噪声比例为 0.1
out1 = sp_noise(img, prob=0.1)
# 添加高斯噪声，均值为0，方差为0.001
out2 = gasuss_noise(img, mean=0, var=0.001)
cv2.imwrite('result/1_.jpg',out2)
cv2.imwrite("result/1__.jpg",out1)

# 2.滤波算法
import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib

'''
算术平均滤波法
'''


def ArithmeticAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


'''
递推平均滤波法
'''


def SlidingAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
中位值平均滤波法
'''


def MedianAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
        mean.append(tmp.mean())
    return mean


'''
限幅平均滤波法
Amplitude:	限制最大振幅
'''


def AmplitudeLimitingAverage(inputs, per, Amplitude):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0]  # 上一次限幅后结果
    for tmp in inputs:
        for index, newtmp in enumerate(tmp):
            if np.abs(tmpnum - newtmp) > Amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
一阶滞后滤波法
a:			滞后程度决定因子，0~1
'''


def FirstOrderLag(inputs, a):
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs


'''
加权递推平均滤波法
'''


def WeightBackstepAverage(inputs, per):
    weight = np.array(range(1, np.shape(inputs)[0] + 1))  # 权值列表
    weight = weight / weight.sum()

    for index, tmp in enumerate(inputs):
        inputs[index] = inputs[index] * weight[index]
    return inputs


'''
消抖滤波法
N:			消抖上限
'''


def ShakeOff(inputs, N):
    usenum = inputs[0]  # 有效值
    i = 0  # 标记计数器
    for index, tmp in enumerate(inputs):
        if tmp != usenum:

            i = i + 1
            if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs


'''
限幅消抖滤波法
Amplitude:	限制最大振幅
N:			消抖上限
'''


def AmplitudeLimitingShakeOff(inputs, Amplitude, N):
    # print(inputs)
    tmpnum = inputs[0]
    for index, newtmp in enumerate(inputs):
        if np.abs(tmpnum - newtmp) > Amplitude:
            inputs[index] = tmpnum
        tmpnum = newtmp
    # print(inputs)
    usenum = inputs[0]
    i = 0
    for index2, tmp2 in enumerate(inputs):
        if tmp2 != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index2] = usenum
    # print(inputs)
    return inputs


T = np.arange(0, 0.5, 1 / 4410.0)
num = signal.chirp(T, f0=10, t1=0.5, f1=1000.0)
pl.subplot(2, 1, 1)
pl.plot(num)
result = ArithmeticAverage(num.copy(), 30)

# print(num - result)
pl.subplot(2, 1, 2)
pl.plot(result)
pl.show()

