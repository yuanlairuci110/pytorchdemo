# encoding=utf-8
import jieba
import jieba.analyse as analyse

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

# 加载停用词
#把停用词做成字典
stopwords = {}
fstop = open('stop_words.txt', 'r',encoding="utf-8")
for eachWord in fstop:
    stopwords[eachWord]=eachWord
fstop.close()
wordList = list(jieba.cut(str))
outStr = ''
for word in wordList:
    if word not in stopwords:
        outStr += word
        outStr += ' '
print("去停用词后的分词 ：",outStr)

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

#=================TF-IDF关键词提取==============
text = '新媒体运营如何提升自己的写作能力'
# 添加新词
word = '新媒体运营'
jieba.suggest_freq((word), True)
jieba.add_word(word, freq=100, tag='get')
# 利用idf进行关键词提取
# jieba.analyse.set_idf_path("/idf.txt.big")
print (' '.join(jieba.analyse.extract_tags(text, topK=100, withWeight=False, allowPOS=('get','n','ns','vn'))))

#==============基于 TextRank 算法的关键词抽取============
