text = "this is  aa  s  你好的啊 \n  you and mim aana 好的"
# print(text)

# 写文件
my_file=open('my file.txt','w')   #用法: open('文件名','形式'), 其中形式有'w':write;'r':read.
my_file.write(text)               #该语句会写入先前定义好的 text
my_file.close()                   #关闭文件

# 追加文件
append_text='\nThis is appended file.'  # 为这行文字提前空行 "\n"
append_file=open('my file.txt','a')   # 'a'=append 以增加内容的形式打开
append_file.write(append_text)
append_file.close()

# 读文件
read_file= open('my file.txt','r')

# 多所有的内容
content=read_file.read()
print(content)
read_file.close()
