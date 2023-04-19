# 算法


# 1.最优二叉搜索树
def Optimal_BST(p,q,n):
    # p->List,q->List,n->int.q为非关键词概率
    # e[1...n+1,0...n]   w[1...n+1,0...n]  root[1...n,1...n]
    length_e=int((n+2)*(n+1)/2)
    length_root=int(n*(n+1)/2)
    x=0
    # 上三角矩阵按列存储,A[i,j]=T[j*(j+1)/2+i]
    e=[x for i in range(length_e)]
    w=[x for i in range(length_e)]
    root=[x for i in range(length_root)]
    for i in range(n+1):
        index=int((i+1)*i/2+i)   #i,j=i
        e[index]=q[i]  #这里伪代码用减一表示从0..n，所以这里要变
        w[index]=q[i]
    for l in range(1,n+1):
        for i in range(n-l+1):
            j=i+l
            index=int(j*(j+1)/2+i)
            e[index]=float('inf')
            w[index]=w[int((j-1)*j/2+i)]+p[j]+q[j]
            for r in range(i,j+1):
                t=e[int(r*(r-1)/2+i)]+e[int(j*(j+1)/2+r)]+w[index]
                if t<e[index]:
                    e[index]=t
                    root_index=int((j-1)*j/2+i)
                    root[root_index]=r
    return [e,w,root]
p=[0,0.15,0.1,0.05,0.1,0.2]
q=[0.05,0.1,0.05,0.05,0.05,0.1]
n=5
[e,w,root]=Optimal_BST(p,q,n)
print([e,w,root].__len__())
print(e)
print(w)
print(root)
print([e,w,root])


print('------------------------------------------------------------------------------------------------------------')
# # 2.排序算法
def quicksort(list,p,r):
    """
    快速排序
    :param list:
    :param p:
    :param r:
    :return:
    """
    if p<r:
        q=partion(list,p,r)
        quicksort(list,p,q)
        quicksort(list,q+1,r)
def partion(list,p,r):
	i=p-1
	for j in range(p,r):
		if list[j]<=list[r]:
			i+=1
			list[i],list[j]=list[j],list[i]
	list[i+1],list[r]=list[r],list[i+1]
	return i
list1=[2,8,7,1,3,5,6,4]
quicksort(list1,0,len(list1)-1)
print (list1)

# 排序过程
def quicksort(list,p,r):
	if p<r:
		q=partion(list,p,r)
		quicksort(list,p,q-1)
		quicksort(list,q+1,r)
def partion(list,p,r):
	i=p-1
	for j in range(p,r):
		if list[j]<list[r]:
			i+=1
			list[i],list[j]=list[j],list[i]
		print('+',list,'+')
	list[i+1],list[r]=list[r],list[i+1]
	return i+1





def busort(list1):

    """

    冒泡排序
    :param list1:
    :return:
    """
    for i in range(len(list1)):
        for i in range(len(list1)-1-i):
            if list1[i] > list1[i+1]:
                list1[i], list1[i+1] = list1[i+1], list1[i]
            
  
    return list1


def chansort(list1):
    """
    最值排序
    :param list1:
    :return:
    """
    for i in range(len(list1)):
        minindex = np.argmin(list1[i:len(list1)])
        # # for j in range(len(list1)-i):
        # temp = list1[i]
        # list1[i] = list1[i:len(list1)][minindex]
        # list1[i:len(list1)][minindex] = temp
        list1[i], list1[i:len(list1)][minindex] = list1[i:len(list1)][minindex], list1[i]
    return list1

def insort(list1):
    """
    插入排序
    :param list1:
    :return:
    """

    for i in range(len(list1)):
        for j in range(i):
            if list1[i] < list1[j]:
               list1.insert(j,list1[i])
               list1.pop(i+1)

    return list1

def minsort(list1):
    """
    缩小增量排序
    :param list1:
    :return:
    """

    gap = len(list1)/2
    while gap >= 1:
        gap = int(gap)
        for j in range(gap, len(list1)):
            i = j
            while (i - gap) >= 0:
                if list1[i] < list1[i - gap]:
                    list1[i], list1[i - gap] = list1[i - gap], list1[i]
                    i -= gap
                else:
                    break
        gap //= 2

    return list1

def merge_sort(array):
    if len(array) == 1:
        return array
    left_array = merge_sort(array[:len(array)//2])
    right_array = merge_sort(array[len(array)//2:])
    return merge(left_array, right_array)


def merge(left_array, right_array):
    left_index, right_index, merge_array = 0, 0, list()
    while left_index < len(left_array) and right_index < len(right_array):
        if left_array[left_index] <= right_array[right_index]:
            merge_array.append(left_array[left_index])
            left_index += 1
        else:
            merge_array.append(right_array[right_index])
            right_index += 1
    merge_array = merge_array + left_array[left_index:] + right_array[right_index:]
    return merge_array


def heap_sort(nums):
    """
    堆排序
    :param nums:
    :return:
    """

    build_heap(nums)
    for i in range(len(nums)-1,-1,-1):
        nums[0], nums[i] = nums[i], nums[0]
        max_heapify(nums, 0, i)

def build_heap(nums):

    lenght = len(nums)
    for i in range((lenght-1)//2,-1,-1):
        max_heapify(nums, i, lenght)

def max_heapify(nums, i, lenght):

    #i父节点的位置，length数组长度
    #找到节点的左右孩子节点
    left = i*2+1
    right = i*2+2
    #判断左右孩子节点与父节点的大小
    if left < lenght and nums[left] > nums[i]:
        largest = left
    else:
        largest = i
    if right < lenght and nums[right] > nums[largest]:
        largest = right
    if largest != i:
        nums[i], nums[largest] = nums[largest], nums[i]
        #调整子树
        max_heapify(nums, largest, lenght)



def counting_sort(array):
    """
    计数排序
    :param array:
    :return:
    """
    if len(array) < 2:
        return array
    max_num = max(array)
    count = [0] * (max_num + 1)
    for num in array:
        count[num] += 1
    new_array = list()
    for i in range(len(count)):
        for j in range(count[i]):
            new_array.append(i)
    return new_array

def bucket_sort(array):
    """
    桶排序
    :param array:
    :return:
    """
    min_num, max_num = min(array), max(array)
    bucket_num = (max_num-min_num)//3 + 1
    buckets = [[] for _ in range(int(bucket_num))]
    for num in array:
        buckets[int((num-min_num)//3)].append(num)
    new_array = list()
    for i in buckets:
        for j in sorted(i):
            new_array.append(j)
    return new_array


def radix_sort(array):
    max_num = max(array)
    place = 1
    while max_num >= 10**place:
        place += 1
    for i in range(place):
        buckets = [[] for _ in range(10)]
        for num in array:
            radix = int(num/(10**i) % 10)
            buckets[radix].append(num)
        j = 0
        for k in range(10):
            for num in buckets[k]:
                array[j] = num
                j += 1
    return array


import numpy as np

list1 = np.random.randint(0,100,10,np.int32)
list1 = list(list1)
# list1.append(0)
# list1=[2,8,7,1,3,5,6,4,0,9,16,28,19,65,45,23,15,48,26]
# quicksort(list1,0,len(list1)-1)
# list2 = busort(list3)
# list2 = chansort(list1)
# list1=[2,8,7,1,3,5,6,4]
# list2 = insort(list1)
# quicksort(list3,0,len(list1)-1)

# list2 =minsort(list1)
# heap_sort(list1)
# print(list1)
# list2 = merge_sort(list1)
list2 = radix_sort(list1)
print(list2)


# 搜索算法
# 搜索算法
import numpy as np


def linear_search(data, search_for):
    """线性搜索"""
    search_at = 0
    search_res = False
    while search_at < len(data) and search_res is False:
        if data[search_at] == search_for:
            search_res = True
        else:
            search_at += 1
    return search_res


lis = [2, 5, 10, 7, 35, 12, 26, 41]
print(linear_search(lis, 12))
print(linear_search(lis, 6))
def insert_search(data,x):
    """插值搜索"""
    idx0 = 0
    idxn = (len(data) - 1)
    while idx0 <= idxn and x >= data[idx0] and x <= data[idxn]:
        mid = idx0 +int(((float(idxn - idx0)/(data[idxn] - data[idx0])) * (x - data[idx0])))
        if data[mid] == x:
            return "在下标为"+str(mid) + "的位置找到了" + str(x)
        if data[mid] < x:
            idx0 = mid + 1
    return "没有搜索到" + str(x)
lis = [2, 6, 11, 19, 27, 31, 45, 121]



print(insert_search(lis, 45))
print(insert_search(lis, 3))


def max_min(lst):
    """
    查找最值
    :param lst:
    :return:
    """
    max = min = lst[0]
    for i in range(len(lst)):
        if lst[i] > max:
            max = lst[i]
            a = i
        if lst[i] < min:
            min = lst[i]
            b = i
    return [[max, min],[a, b]]


alist = [5, 3, 7, 2, 12, 45, 16, 23]  # 测试列表
print("列表中元素的(最大值,最小值)=", max_min(alist))  # 查找列表的最大值最小值


def binary_search(lst, key):
    low = 0  # 左边界
    high = len(lst) - 1  # 右边界
    time = 0  # 记录查找次数
    while low <= high:  # 左边界小于等于右边界，则循环
        time += 1
        mid = (low + high) // 2
        if lst[mid] > key:  # 中间位置元素大于要查找的值
            high = mid - 1
        elif lst[mid] < key:
            low = mid + 1
        else:
            print("折半查找的次数: %d" % time)
            print("所要查找的值在列表中的索引号是：", end='')
            return mid
    print("折半查找的次数: %d" % time)
    return '未找到'  # 查找不成功，'未找到'



alist = [5, 13, 19, 21, 37, 56, 64, 75, 80, 88, 92]
result = binary_search(alist, 13)
print(result)


def binary_search(lst, low, high, key):
    mid = int((low + high) / 2)
    if high < low:
        return False
    elif lst[mid] == key:
        print("所要查找的值在列表中的索引号是：", end='')
        return mid
    elif lst[mid] > key:
        high = mid - 1
        return binary_search(lst, low, high, key)
    else:
        low = mid + 1
        return binary_search(lst, low, high, key)



alist = [5, 13, 19, 21, 37, 56, 64, 75, 80, 88, 92]
low = 0
high = len(alist) - 1
result = binary_search(alist, 0, high, 21)
print(result)


def interpolation_search(lst, low, high, key):
    time = 0  # 用来记录查找次数
    while low < high:
        time += 1
        # 计算mid值是插值算法的核心代码
        mid = low + int((high - low) * (key - lst[low]) / (lst[high] - lst[low]))
        print("mid={0}, low={1}, high={2}".format(mid, low, high))
        if lst[mid] > key:
            high = mid - 1
        elif lst[mid] < key:
            low = mid + 1
        else:
            print("插值查找%s的次数:%s" % (key, time))  # 打印插值查找的次数
            return mid
    print("插值查找%s的次数:%s" % (key, time))
    return False



alist = [5, 13, 19, 21, 37, 56, 64, 75, 80, 88, 92]
low = 0
high = len(alist) - 1
interpolation_search(alist, low, high, 13)





