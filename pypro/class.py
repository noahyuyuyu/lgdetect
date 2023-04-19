# class & object


# 1.直接定义类，联结对象

class People(object):
    def __init__(self,name,age,gender):
        self.name = name
        self.age = age
        self.gender = gender
    def huiJia(self):
        print("%s,%d,%s,辍学回家" %(self.name,self.age,self.gender))
    def quXiFu(self):
        print("%s,%d,%s,开车去娶媳妇" %(self.name,self.age,self.gender))
    def kanChai(self):
        print("%s,%d,%s,上山砍柴" %(self.name,self.age,self.gender))
Laoli = People('老李',18,'男')
zhangsan = People('校思浩',22,'男')
lisi = People('唐浩',10, '女')

Laoli.quXiFu()
zhangsan.kanChai()
lisi.huiJia()

print("*" * 20)
# 2.类封装栈，导出对象
#class Stack:
    # 栈的方法:
    # 入栈(push), 出栈(pop), 栈顶元素(top),
    # 栈的长度(lenght), 判断栈是否为空(isempty)
    # 显示栈元素(view)
    # 操作结果:
    # 栈类的实例化
    # 入栈2次
    # 出栈1次
    # 显示最终栈元素

class Stack(object):
    # 构造函数
    def __init__(self):
        self.stack = []
    def push(self, value):
        """
        :param value: 入栈元素
        :return:
        """
        self.stack.append(value)
        return True

    def pop(self):
        # 判断栈是否为空
        if self.stack:
            # 获取出栈元素, 并返回
            item = self.stack.pop()
            return  item
        else:
            return  False

    def top(self):
        if self.stack:
            return  self.stack[-1]
        else:
            return  False
    def length(self):
        return  len(self.stack)

    def isempty(self):
        return self.stack==[]

    def view(self):
        return  ",".join(self.stack)

s = Stack()
s.push('1')
s.push('2')
s.push('3')
s.push('4')
print(s.top())
print(s.length())
print(s.isempty())
s.pop()
print(s.view())

print("*" * 20)
# 3.类封装队列，导出对象

class Queue(object):
    # 构造函数
    def __init__(self):
        self.queue = []
    def push(self, value):
        self.queue.append(value)
        return True
    def pop(self):
        if self.queue:
            del self.queue[-1]
        else:
            return  False

    def front(self):
        if self.queue:
            return  self.queue[0]
        else:
            return  False
    def rear(self):
        if self.queue:
            return  self.queue[-1]
        else:
            return  False
    def length(self):
        return  len(self.queue)

    def isempty(self):
        return self.queue==[]

    def view(self):
        return  ",".join(self.queue)
s = Queue()
s.push('1')
s.push('2')
s.push('3')
s.push('4')
print(s.front())
print(s.rear())
print(s.length())
print(s.isempty())
s.pop()
print(s.view())

print("*" * 20)
# 4.类的继承

class Animals(object):
    def __init__(self, name, age):
        self.name = name
        self.age= age
    def eat(self):
        print('eating......')


class Dog(Animals):  # 当Dog没有构造方法时，执行Animals里面的构造方法
    def __init__(self, name, age, power):
        # self.name = name
        # self.age = age
        # 执行Dog的父类的构造方法;
        super(Dog, self).__init__(name, age)
        self.power = power
    def eat(self):
        print(self.power)
        super(Dog, self).eat()


#  1. 如果子类没有的属性和方法， 则去父类找， 如果父类也没有， 就报错。
d1 = Dog("大黄",3,100)
print(d1.name)
print(d1.age)
print(d1.power)
d1.eat()

print("*" * 20)
# 5. 类的多继承

class D:
    def test(self):
        print("D test")
class C(D):
    pass
    def test(self):
        print("C test")
class B(D):
    pass
    # def test(self):
    #     print("B test")
class A(B,C):
    pass
    # def test(self):
    #     print("A test")
a = A()
a.test()

print("*" * 20)
print('------------------------------------')
# 6.类的多态性
# 子类和父类存在相同方法时，子类会覆盖父类方法
# 运形时总会调用子类方法--> 多态


class Animal(object):
    def run(self):
        print('running...')
    def cry(self):
        print('crying...')

class Dog(Animal):
    def run(self):
        print('dog running...')

    def eat(self):
        print('dog eating...')

class Cat(Animal):
    def run(self):
        print('cat running...')

cat = Cat()
cat.run()

dog = Dog()
dog.run()
