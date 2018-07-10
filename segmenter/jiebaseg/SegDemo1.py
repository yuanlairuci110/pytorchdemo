# encoding=utf-8
import jieba

# ==========================分词=====================

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode  全模式 : " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode  精确模式: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

#=========================自定义字典=================

# 未加载自定义字典前
str = "李小福是创新办主任也是云计算方面的专家，刘诗诗她只是一个演员假"
seg_list = jieba.cut(str)
print("未加载字典 ： "+", ".join(seg_list))

# 加载自定义字典
jieba.load_userdict("userdict.txt")
seg_list = jieba.cut(str)
print("加载字典 ："+", ".join(seg_list))

# 程序添加自定义词典词
jieba.add_word(("刘诗诗"))
jieba.add_word(("演员假"))
seg_list = jieba.cut(str)
print("加载程序字典 ："+", ".join(seg_list))

# 调节单个词语的词频
jieba.suggest_freq(('员', '假'), False)
seg_list = jieba.cut(str)
print("加载程序字典 ："+", ".join(seg_list))
jieba.suggest_freq(('演', '员'), True)
seg_list = jieba.cut(str)
print("加载程序字典 ："+", ".join(seg_list))

