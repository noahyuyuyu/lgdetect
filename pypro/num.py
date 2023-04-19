# 数值分析



# 当不确定将多少个参数传递给函数，或者我们想要将存储的列表或参数元组传递给函数时，我们使用*args。
# 当不知道将多少关键字参数传递给函数时使用**kwargs，或者它可以用于将字典的值作为关键字参数传递。
# 标识符*args和**kwargs是一个约定，也可以使用* bob和** billy。
# 迭代器是可以遍历或迭代的对象，返回可迭代项集的函数称为生成器。
# int（） - 将任何数据类型转换为整数类型
# float（） - 将任何数据类型转换为float类型
# ord（） - 将字符转换为整数
# hex（） - 将整数转换为十六进制
# oct（） - 将整数转换为八进制
# tuple（） -此函数用于转换为元组。
# set（） -此函数在转换为set后返回类型。
# list（） -此函数用于将任何数据类型转换为列表类型。
# dict（） -此函数用于将顺序元组（键，值）转换为字典。
# str（） -用于将整数转换为字符串。
# complex（real，imag）- 此函数将实数转换为复数（实数，图像）数。

from numpy import mat
import numpy as np
import torch

# 1.数据类型

tuplel_demo = ('hello', 100, 4.5) #元组
print(tuplel_demo[0])
print(tuplel_demo[1])
print(tuplel_demo[2])





print('------------------------------------')
# 2.运算符

x = np.array([0, 1, 2, 3, 4])    # 等价于:x=np.arange(0,5)
y = x[::-1]
print(x)
print(y)
print(np.dot(x, y))


a = np.array([1,2,3])
b = np.eye(3)
c = np.ones(3)
d = np.mat('1;4;7')

e = np.multiply(a,b)#点乘
f=c*d
g=c.dot(d)

print(a,b,c,d,e,f,g)






A = torch.tensor([1,1,1])
B = torch.tensor([1,2,3])
C = torch.tensor(d)






print(A,B,C)
print(torch.mul(A,B))#哈达玛积
print(torch.matmul(A,B))

print('------------------------------------')

# 3. 索引
# [::-1] 顺序相反操作
# [-1] 读取倒数第一个元素
a = 'python'
b = a[::-1]
print(b)  # nohtyp

c = a[::-2]
print(c)  # nhy

# 从后往前数的话，最后一个位置为-1
d = a[:-1]  # 从位置0到位置-1之前的数

print(d)  # pytho

e = a[:-2]  # 从位置0到位置-2之前的数

print(e)  # pyth

a = [0,1,2,3,4,5,6,7,8,9]
b = a[1:3]   # [1,2]
print(b)

S = 'abcdefg'
print(S[0],S[-1])
#('a', 'g')

#张量

A=np.array([[1,23,6,56,56,26,2,262,26]])
B=A[:,1:5]
C=A[:,[1,6]]
print(B,C)