# 边缘检测算法
# 图像边缘信息主要集中在高频段，通常说图像锐化或检测边缘，实质就是高频滤波。
# 我们知道微分运算是求信号的变化率，具有加强高频分量的作用。
# 在空域运算中来说，对图像的锐化就是计算微分。对于数字图像的离散信号，微分运算就变成计算差分或梯度。
# 图像处理中有多种边缘检测（梯度）算子，常用的包括普通一阶差分，Robert算子（交叉差分），Sobel算子等等，是基于寻找梯度强度。
# 拉普拉斯算子（二阶差分）是基于过零点检测。通过计算梯度，设置阀值，得到边缘图像。

# 1.Sobel算子
# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("D:\MDT\code\pypro\data/net.png", 0)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.Canny算子
# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("D:\MDT\code\pypro\data/net.png", 0)

img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。
canny = cv2.Canny(img, 50, 150)  # 最大最小阈值

cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3.Laplacian算子
# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("D:\MDT\code\pypro\data/net.png", 0)

gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(gray_lap)

cv2.imshow('laplacian', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

