# coding:utf8
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from com.charrnn.model import PoetryModel
import tqdm


# 定义超参数类
class Config(object):
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 1
    batch_size = 128
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20  # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env = 'poetry'  # visdom env
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = None  # 预训练模型路径
    prefix_words = '细雨鱼儿出,微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/tang'  # 模型保存路径


opt = Config()


# 打印一首歌
def load_one_poem():
    # 加载数据
    datas = np.load('tang.npz')
    data = datas['data']
    ix2word = datas['ix2word'].item()

    # 查看第一首歌
    poem = data[1]

    # 词序号转成对应的汉字
    poem_txt = [ix2word[ii] for ii in poem]

    # 显示
    print(''.join(poem_txt))


# 打印一首歌
# load_one_poem()

# 训练模型
def train():
    # 获取数据
    datas = np.load(opt.pickle_path)
    data, word2ix, ix2word = datas['data'], datas['word2ix'].item(), datas['ix2word'].item()
    # print(ix2word)

    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=2)
    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练数据
    for epoch in range(opt.epoch):
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous()
            optimizer.zero_grad()
            input_, target = Variable(data_[:-1, :]), Variable(data_[1:, :])
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            if ii % 200 == 0:
                print("\nii : ",ii,"保存模型")
                torch.save(model, '%s_%s.pkl' % (opt.model_prefix, epoch))
                torch.save(model.state_dict(), '%s_%s_params.pkl' % (opt.model_prefix, epoch))

        torch.save(model, '%s_%s.pkl' % (opt.model_prefix, epoch))
        torch.save(model.state_dict(), '%s_%s_params.pkl' % (opt.model_prefix, epoch))


if __name__ == '__main__':

    # 训练
    train()


