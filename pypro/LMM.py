# 拉格朗日乘数法（Lagrange Multiplier Method）
# 在椭球条件下，求解 f（x，y，z）=8xyz的最大值
from scipy.optimize import minimize
import numpy as np
e = 1e-10 # 非常接近0的值
fun = lambda x : 8 * (x[0] * x[1] * x[2]) # f(x,y,z) =8 *x*y*z
cons = ({'type': 'eq', 'fun': lambda x: x[0]**2+ x[1]**2+ x[2]**2 - 1}, # x^2 + y^2 + z^2=1
        {'type': 'ineq', 'fun': lambda x: x[0] - e}, # x>=e等价于 x > 0
        {'type': 'ineq', 'fun': lambda x: x[1] - e},
        {'type': 'ineq', 'fun': lambda x: x[2] - e}
       )
x0 = np.array((1.0, 1.0, 1.0)) # 设置初始值
res = minimize(fun, x0, method='SLSQP', constraints=cons)
print('最大值：',res.fun)
print('最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)

print("*******************************")
# python符号运算

from sympy import symbols
from scipy import poly1d

# 定义符号
x, p1, p2, p3 = symbols('x p1 p2 p3')

# 一次定义多个符号,返回tuple
vrs = symbols('p:5')  # (p0, p1, p2, p3, p4)
vrs = symbols('p1:5')  # (p1, p2, p3, p4)

# 循环定义多组符号
for k in range(2, 12):
    multi_vrs = symbols('p1:' + str(k))


# 1)用poly1d生成多项式，可拥有多项式任何属性，如获取系数、获取x为某个值时的结果等，
# 虽然可以带入另一个多项式的x，但不能print(G)
# ρ = ploy1d(list(vrs))
# print(ρ)  # p0*x**4 + p1*x**3 + p2*x**2 + p3*x + p4
# print(ρ.c)  # 获取系数
# print(ρ(1))  # x = 1时多项式的系数

# 2)自己定义多项式， sum运用了符号的加法运算来生成多项式
def polyn(x, vrs):
    n = len(vrs)
    result = 0 + sum([v * x ** i for i, v in enumerate(vrs)])
    return result


e = 0.4  # constant
l = [0, 0, 0, 1, 0, 0]  # 假设λ多项式的系数为l3 = 1, 其他系数为0
λ = poly1d(l)  # λ=x**2 (l3 = 1)
ρ = polyn(x, vrs)
G = 1 - e * λ(1 - ρ) - x
print(G)  # 1 - 0.4*(-p1 - p2*x - p3*x**2 - p4*x**3 + 1)**2

print("*******************************")
# python嵌套优化问题
import numpy as np
from scipy.optimize import fminbound, minimize


# objective function: f
def f(l):
    n = len(l)
    l = [l[i] / (i + 1) for i in range(n)]  # pi / i, i = 1...n
    return sum(l)  # 注意加上负号转换为求最小值！！！


# inner_cons
def cons(l):
    # 用outer_minimize的变量l作为系数定义inner_minimize
    # 这里的多项式就是对应上面符号运算的结果，只把p换成l
    # G=1 - 0.4*(-p0*x**4 - p1*x**3 - p2*x**2 - p3*x - p4 + 1)**2
    funbnd = lambda x: 1 - 0.4 * (-l[0] - l[1] * x - l[2] * x ** 2 - l[3] * x ** 3 + 1) ** 2 - x

    bnds2 = ((1 - e, 1),)

    # optimize f under cons
    res = minimize(funbnd, x0=(1 - e,), method='SLSQP', bounds=bnds2)
    # print("cons2:", res.fun)
    # print(res)
    return res.fun


# outer_cons
# * ineq means to be non-negative, eq means to be zero
Cons = (
    {'type': 'ineq', 'fun': cons},  # inner_cons

    {'type': 'eq', 'fun': lambda l: sum(l) - 1}  # l1 + l2 + ... + ln = 1
)

# initial value of variables：array, tuple or list
initial_x = np.array([0 for _ in range(4)])
# initial_x = (0, 0, 0, 0)
# initial_x = [0, 0, 0, 0]


# bounds： array, tuple or List
bnds = ((0, 1), (0, 1), (0, 1), (0, 1))

# outer_ominimize f under Cons
res = minimize(f, x0=initial_x, method='SLSQP', bounds=bnds, constraints=Cons)
print(res)

print("*******************************")

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fminbound, minimize
from scipy import poly1d
from sympy import symbols, lambdify

np.set_printoptions(precision=3, suppress=True,
                    formatter={'float': '{: 0.3f}'.format})  # turn off np’s scientific notation

e = 0.4
x = symbols('x')


def polyn(x, vrs, length):
    # order : p1 x^1 + p2 x^2 + p3 x^3 + pn x^n
    result = 0 + sum([vrs[i] * x ** i for i in range(length)])
    return result


def find_min(G, p, l, var_n):
    # inner_cons: method1
    # def cons(l):
    #     new_G = G.subs([(p[i], l[i]) for i in range(var_n)])
    #     f = lambdify(x, new_G, 'numpy')
    #     bnds2 = ((1 - e, 1),)
    #     # inner optimize f under inner cons
    #     res1 = minimize(f, x0=((2 - e) / 2,), method='SLSQP', bounds=bnds2)
    #     res2 = minimize(f, x0=(1 - e,), method='SLSQP', bounds=bnds2)
    #     res3 = minimize(f, x0=(1,), method='SLSQP', bounds=bnds2)
    #     # print(res.fun, res.x)
    #     # print(res.message)
    #     return min(res1.fun, res2.fun, res3.fun)

    # inner_cons: method2
    def cons(l):
        new_G = G.subs([(p[i], l[i]) for i in range(var_n)])
        f = lambdify(x, new_G, 'numpy')
        sample_point = np.linspace(1 - e, 1, num=1000)
        # min = 0
        y = f(sample_point)
        return min(y)

    # outer_cons
    # * ineq means to be non-negative, eq means to be zero
    Cons = (
        {'type': 'ineq', 'fun': cons},  # G_min >= 0

        {'type': 'eq', 'fun': lambda l: sum(l) - 1}  # l1 + l2 + ... + ln = 1
    )

    # objective function: f
    def f(l):
        return sum([l[i] / (i + 1) for i in range(len(l))])

    # initial guess
    initial_x = np.array([1 / var_n for _ in range(var_n)])

    # bounds
    bnds = [(0, 1) for _ in range(var_n)]

    # outer optimize f under outer Cons
    # f(l), l = [l[0], l[1], l[2], l[3],...]
    res = minimize(f, x0=initial_x, method='SLSQP', bounds=bnds, constraints=Cons)  # SLSQP, Newton-CG, 'BFGS'
    print(res)

    sum_ρ = res.fun
    sum_λ = f(l)
    rate = 1 - sum_ρ / sum_λ
    print(rate)

    return rate


if __name__ == '__main__':
    # λ(x)
    # l = [0.000,  0.417,  0.068,  0.117,  0.268,  0.108,  0.022]
    l = [0, 0, 1, 0, 0, 0]
    λ = polyn(x, l, len(l))
    λ_func = lambdify(x, λ, 'numpy')

    total_p_len = 20
    p = symbols(f"p1:{total_p_len + 1}")

    max_rate = 0
    j = 0
    for i in range(7, 13):  # len = 7~12
        ρ = polyn(x, p, i)
        # print(ρ)
        # constraint expression
        G = 1 - e * λ_func(1 - ρ) - x
        print(G)
        res = find_min(G, p, l, i)
        if res > max_rate:
            max_rate = res
            j = i
    print(j, max_rate)