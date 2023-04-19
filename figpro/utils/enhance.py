import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


img = cv.imread("../data\images/sky.jpg", 0)
grayHist = calcGrayHist(img)
x = np.arange(256)
# 绘制灰度直方图
plt.plot(x, grayHist, 'r', linewidth=2, c='black')
plt.xlabel("gray Label")
plt.ylabel("number of pixels")
plt.show()
cv.imshow("img", img)
cv.waitKey()


# Matplotlib本身计算直方图的函数hist

img = cv.imread("../data\images/sun.jpg", 0)
h, w = img.shape[:2]
pixelSequence = img.reshape([h * w, ])
numberBins = 256
histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                  facecolor='black', histtype='bar')
plt.xlabel("gray label")
plt.ylabel("number of pixels")
plt.axis([0, 255, 0, np.max(histogram)])
plt.show()
cv.imshow("img", img)
cv.waitKey()


#  线性变换

a = np.array([[0, 200], [23, 4]], np.uint8)
b = 2 * a
print(b.dtype)
print(b)

a = np.array([[0, 200], [23, 4]], np.uint8)
b = 2.0 * a
print(b.dtype)
print(b)



# 灰度直方图函数
def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()


img = cv.imread("../data\images/sun.jpg", 0)
out = 2.0 * img
# 进行数据截断，大于255的值截断为255
out[out > 255] = 255
# 数据类型转换
out = np.around(out)
out = out.astype(np.uint8)
# 分别绘制处理前后的直方图
# grayHist(img)
# grayHist(out)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()

img = cv.imread("../data\images/sky.jpg", 0)
img = cv.resize(img, None, fx=0.3, fy=0.3)
h, w = img.shape[:2]
out = np.zeros(img.shape, np.uint8)
for i in range(h):
    for j in range(w):
        pix = img[i][j]
        if pix < 50:
            out[i][j] = 0.5 * pix
        elif pix < 150:
            out[i][j] = 3.6 * pix - 310
        else:
            out[i][j] = 0.238 * pix + 194
        # 数据类型转换
out = np.around(out)
out = out.astype(np.uint8)
# grayHist(img)
# grayHist(out)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()

# 直方图正规化

img = cv.imread("../data\images/sky.jpg", 0)
# 计算原图中出现的最小灰度级和最大灰度级
# 使用函数计算
Imin, Imax = cv.minMaxLoc(img)[:2]
# 使用numpy计算
# Imax = np.max(img)
# Imin = np.min(img)
Omin, Omax = 0, 255
# 计算a和b的值
a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * img + b
out = out.astype(np.uint8)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()

img = cv.imread("../data\images/sun.jpg", 0)
out = np.zeros(img.shape, np.uint8)
cv.normalize(img, out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()

img = cv.imread("../data\images/sky.jpg", 0)
# 图像归一化
fi = img / 255.0
# 伽马变换
gamma = 0.4
out = np.power(fi, gamma)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()

# 全局直方图均衡化

def equalHist(img):
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


img = cv.imread("../data\images/sky.jpg", 0)
# 使用自己写的函数实现
equa = equalHist(img)
# grayHist(img, equa)
# 使用OpenCV提供的直方图均衡化函数实现
# equa = cv.equalizeHist(img)
cv.imshow("img", img)
cv.imshow("equa", equa)
cv.waitKey()

img = cv.imread("../data\images/sun.jpg", 0)
img = cv.resize(img, None, fx=0.5, fy=0.5)
# 创建CLAHE对象
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 限制对比度的自适应阈值均衡化
dst = clahe.apply(img)
# 使用全局直方图均衡化
equa = cv.equalizeHist(img)
# 分别显示原图，CLAHE，HE
cv.imshow("img", img)
cv.imshow("dst", dst)
cv.imshow("equa", equa)
cv.waitKey()

