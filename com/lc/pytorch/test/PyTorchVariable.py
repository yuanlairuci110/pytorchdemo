# Pytorch （变量）
import torch
from torch.autograd import Variable

tensors = torch.Tensor([[1,2],[3,4]])
print(tensors)
# requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensors,requires_grad = True)
print(variable)

t_out = torch.mean(tensors*tensors)
print(t_out)
v_out = torch.mean(variable*variable)
print(v_out)
v_out.backward()
print(variable.grad)
# print("------反向传递------")
# print(tensors)
print(variable)
print("variable 的tensor 形式：\n",variable.data)    # tensor 形式