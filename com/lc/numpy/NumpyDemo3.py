import numpy as np

a = np.array([10,20,30,40])

b = np.arange(4)

c = a -b

print("a -b : ",c)

c = a + b

print("a + b : ",c)

c = a*b

print("a * b : ",c)

c = b**2

print("b 的平方： ",c)

c = np.sin(a)

print("a 的sin : ",c)

print("逻辑判断 ： ",b<3)

d = np.array([[1,1],[0,1]])

e = np.arange(4).reshape((2,2))

f_doc = np.dot(d,e)

f_doc2 = d.dot(e)

print(f_doc)

print(f_doc2)

h=np.random.random((2,4))
print(h)

# 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元。
print("sum =",np.sum(h,axis=1))

print("min =",np.min(h,axis=0))

print("max =",np.max(h,axis=1))