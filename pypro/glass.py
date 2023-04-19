import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")

## 读取数据集
glass = pd.read_csv(target_url, header=None, prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
print(glass.head())

## 数据集统计
summary = glass.describe()
print(summary)
ncol1 = len(glass.columns)

## 去掉Id列
glassNormalized = glass.iloc[:, 1:ncol1]
ncol2 = len(glassNormalized.columns)
summary2 = glassNormalized.describe()
print(summary2)

## 归一化
for i in range(ncol2):
    mean = summary2.iloc[1, i]
    sd = summary2.iloc[2, i]
    glassNormalized.iloc[:, i:(i + 1)] = (glassNormalized.iloc[:, i:(i + 1)] - mean) / sd

## 绘制箱线图
array = glassNormalized.values
boxplot(array)
plot.xlabel("Attribute Index")
plot.ylabel(("Quartile Ranges - Normalized "))
show()

print('******************************')
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")

## 读取数据集
glass = pd.read_csv(target_url, header=None, prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

glassNormalized = glass
ncols = len(glassNormalized.columns)
nrows = len(glassNormalized.index)
summary = glassNormalized.describe()
nDataCol = ncols - 1

## 归一化
for i in range(ncols - 1):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    glassNormalized.iloc[:, i:(i + 1)] = (glassNormalized.iloc[:, i:(i + 1)] - mean) / sd

for i in range(nrows):
    # plot rows of data as if they were series data
    dataRow = glassNormalized.iloc[i, 1:nDataCol]
    labelColor = glassNormalized.iloc[i, nDataCol] / 7.0
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()


print('******************************')
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")

## 读取数据集
glass = pd.read_csv(target_url,header=None,prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

## 计算所有实值列（包括目标）的相关矩阵
corMat = DataFrame(glass.iloc[:,1:-1].corr())
print(corMat)

## 使用热图可视化相关矩阵
plot.pcolor(corMat)
plot.show()
