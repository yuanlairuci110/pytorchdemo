# Pytorch 基础（张量）

# 导入torch基本的函数
import torch

# 导入torch自动求导函数
import torch.autograd as autograd  # 自动求导

# 导入torch 神经网络的函数
import torch.nn as nn

# 导入torch 激励函数
import torch.nn.functional as F

# 导入 torch 优化函数
import torch.optim as optim

# 导入 numpy
import numpy as np

print("----------创建Tensors(张量)--------")

print("1维的张量")
v_list = [1, 2, 3]
v_tensors = torch.Tensor(v_list)
print(v_tensors)

np_1v_data = np.arange(4).reshape(4)
print("通过numpy 创建一个1维 :", np_1v_data)
tensors_1v_data = torch.from_numpy(np_1v_data)
print("1维numpy 转 tensors : ", tensors_1v_data)
numpy_1v_data = tensors_1v_data.numpy()
print("1维tensors 转numpy : ", numpy_1v_data)

print("2维的张量")
v2_list = [[1, 2, 3], [4, 5, 6]]
v2_tensors = torch.Tensor(v2_list)
print(v2_tensors)

np_2v_data = np.arange(10).reshape(2, 5)
print("通过numpy 创建一个2维 :", np_2v_data)
tensors_2v_data = torch.from_numpy(np_2v_data)
print("2维numpy 转 tensors : ", tensors_2v_data)
numpy_2v_data = tensors_2v_data.numpy()
print("2维tensors 转numpy : ", numpy_2v_data)

print("3维的张量")
v3_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
v3_tensors = torch.Tensor(v3_list)
print(v3_tensors)

print("----获取张量的值---")
print(v_tensors[0])
print(v2_tensors[0])
print(v3_tensors[0])

print("------产生随机数------")
random_tensors = torch.randn((2, 6))
print(random_tensors)

print("------tensors 计算-------(API地址 https://pytorch.org/docs/stable/torch.html)")
x = torch.Tensor([0.1, 0.11, 11])
y = torch.Tensor([0.2, 0.22, 22])
z = x + y
print(z)

# API 接口地址 ： https://pytorch.org/docs/stable/torch.html

data = [[1, 2], [3, 4]]
tensors_data = torch.FloatTensor(data)
numpy_data = np.array(data)
print("\nnumpy相乘:", np.matmul(data,data),
      "\nnumpy相乘另一种方式：",numpy_data.dot(numpy_data),
      "\ntorch相乘：",torch.mm(tensors_data,tensors_data))
# 新版本中(>=0.3.0), 关于 tensor.dot() 有了新的改变, 它只能针对于一维的数组.
print("tensors dot :",torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1])))

print("-----------Tensor维度变型reshaping----------")
x = torch.randn(2, 3, 4)
print("变形前\n",x)
print("变形后\n",x.view(2,12))#将234 -> 2*12