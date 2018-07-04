import warnings

'''
配置文件
'''


class DefaultConfig(object):
    train_data_root = 'D:/python/data/dogvscat/train/'  # 训练集存放路径
    test_data_root = 'D:/python/data/dogvscat/test1/'  # 测试集存放路径
    lc_data_root = 'D:/python/data/dogvscat/test/'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    batch_size = 32  # batch size
    num_workers = 4  # how many workers for loading data


opt = DefaultConfig()
