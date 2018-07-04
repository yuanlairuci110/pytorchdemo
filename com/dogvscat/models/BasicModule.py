# 基础模型
import torch
from torch import nn
import time


class BasicModule(nn.Module):
    """
    主要提供 save和load方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        # 模型默认名字
        self.model_name = str(type(self))

    def load(self, path):
       """
       可加载指定路径的模型
       :param path:
       :return:
       """
       self.load_state_dict(torch.load(path))

    def save(self,name = None):
        """
        保存模型，默认使用 “模型名字+时间”做为文件名
        如 ：AlexNet_0710_23:57:29.pkl
        :param name:
        :return: 文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pkl')

        torch.save(self.state_dict(),name)
        return name