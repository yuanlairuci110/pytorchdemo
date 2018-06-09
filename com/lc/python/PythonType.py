# python 四中结构

# tuple 类型
print("------tuple 类型------")
tup = ('python', 2.7, 64)
for i in tup:
    print(i)

# dictionary 类型
print("------dictionary 类型------")
dic = {}
dic['lan'] = 'python'
dic['version'] = 2.7
dic['platform'] = 64
for key in dic:
    print("KEY is : 【",key,"】 value is :【 ",dic[key],"】")

# set 类型
print("--------set 类型--------")
s = set(['python', 'python2', 'python3','python'])
for item in s:
    print(item)

# 迭代器
print("----迭代器 --------")
# define a Fib class
class Fib(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()

# using Fib object
for i in Fib(5):
    print(i)

# 生成器
print("------- 生成器 ---------")
def fib(max):
    a, b = 0, 1
    while max:
        r = b
        a, b = b, a+b
        max -= 1
        yield r

# using generator
for i in fib(5):
    print(i)

# 三目运算
# var = var1 if condition else var2
# 可以这么理解上面这段语句，如果 condition 的值为 True, 那么将 var1 的值赋给 var;如果为 False 则将 var2 的值赋给 var。
print("--------三目操作符--------")
worked = True
result = 'done' if worked else 'not yet'
print(result)