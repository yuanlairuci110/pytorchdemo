import numpy as np

a = np.array([1,23,4],dtype = np.int)

print(a)

print(a.dtype)

b = np.array([[[1,2,3],[4,5,6]],[[7.8,9],[10,11,12]]])

print(b)

print(b.shape)

print(b.size)

c = np.zeros((3,4))

print(c)

d = np.ones((5,2))

print(d)

e = np.arange(10,20,2)

print(e)

f = np.arange(12).reshape((3,4))

print(f)

g = np.linspace(1,10,6).reshape((2,3))

print(g)

# 创建全空数组, 其实每个值都是接近于零的数:
h = np.empty((3,4)) # 数据为empty，3行4列

print(h)