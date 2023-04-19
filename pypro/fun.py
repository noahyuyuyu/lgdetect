# function & module

# 1. 函数定义与使用
# 定义一个函数，能够完成打印信息的功能
def printInfo():
    print('------------------------------------')
    print('         人生苦短，我用Python')
    print('------------------------------------')

# 定义完函数后，函数是不会自动执行的，需要调用它才可以
printInfo()

# 定义一个函数
def test(a,b):
    #"2个数求和"
    print("%d" % (a + b))

test(11,12)

print('------------------------------------')
#2 函数参数
def add2num(a, b):
    c = a+b
    print(c)

add2num(110, 22) # 调用带有参数的函数时，需要在小括号中，传递数据

def printinfo(name, age=35):
   # 打印任何传入的字符串
   print("name: %s" % name)
   print("age %d" % age)

# 调用printinfo函数
printinfo(name="miki")  # 在函数执行过程中 age去默认值35
printinfo(age=9 ,name="miki")

def fun(a, b, *args, **kwargs):

    print("a =%d" % a)
    print("b =%d" % b)
    print("args:")
    print(args)
    print("kwargs: ")
    for key, value in kwargs.items():
        print("key=%s" % value)

fun(1, 2, 3, 4, 5, m=6, n=7, p=8)  # 注意传递的参数对应

def sum_nums_3(a, *args, b=22, c=33, **kwargs):
    print(a)
    print(b)
    print(c)
    print(args)
    print(kwargs)

sum_nums_3(100, 200, 300, 400, 500, 600, 700, b=1, c=2, mm=800, nn=900)

print('------------------------------------')
# 3.函数返回值
# return后面可以是元组，列表、字典等，只要是能够存储多个数据的类型，就可以一次性返回多个数据

def create_nums(num):
    print("---1---")
    if num == 100:
        print("---2---")
        return num + 1  # 函数中下面的代码不会被执行，因为return除了能够将数据返回之外，还有一个隐藏的功能：结束函数
    else:
        print("---3---")
        return num + 2
    print("---4---")


result1 = create_nums(100)
print(result1)  # 打印101
result2 = create_nums(200)
print(result2)  # 打印202
print('------------------------------------')

# 4.函数嵌套
# 写一个函数求三个数的和
# 写一个函数求三个数的平均值
# 求3个数的和
def sum3Number(a,b,c):
    return a+b+c # return 的后面可以是数值，也可是一个表达式

# 完成对3个数求平均值
def average3Number(a,b,c):

    # 因为sum3Number函数已经完成了3个数的就和，所以只需调用即可
    # 即把接收到的3个数，当做实参传递即可
    sumResult = sum3Number(a,b,c)
    aveResult = sumResult/3.0
    return aveResult

# 调用函数，完成对3个数求平均值
result = average3Number(11,2,55)
print("average is %d"%result)

# 5.变量
g_num = 0

def test1():
    global g_num
    # 将处理结果存储到全局变量g_num中.....
    g_num = 100

def test2():
    # 通过获取全局变量g_num的值, 从而获取test1函数处理之后的结果
    print(g_num)

# 1. 先调用test1得到数据并且存到全局变量中
test1()

# 2. 再调用test2，处理test1函数执行之后的这个值
test2()

def test1():
    # 通过return将一个数据结果返回
    return 50

def test2(num):
    # 通过形参的方式保存传递过来的数据，就可以处理了
    print(num)

# 1. 先调用test1得到数据并且存到变量result中
result = test1()

# 2. 调用test2时，将result的值传递到test2中，从而让这个函数对其进行处理
test2(result)

def test1():
    # 通过return将一个数据结果返回
    return 20

def test2():
    # 1. 先调用test1并且把结果返回来
    result = test1()
    # 2. 对result进行处理
    print(result)

# 调用test2时，完成所有的处理
test2()

