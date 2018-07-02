# coding:utf8
import numpy as np

# 加载数据
datas = np.load('tang.npz')
data = datas['data']
ix2word = datas['ix2word'].item()

# 查看第一首歌
poem = data[0]

# 词序号转成对应的汉字
poem_txt = [ix2word[ii] for ii in poem]

# 显示
print(''.join(poem_txt))