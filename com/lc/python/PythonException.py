try:
    file=open('eeee.txt','r+') #会报错的代码
except Exception as e: # 将报错存储在 e 中
    print(e)
    response = input('do you want to create a new file:')
    if response=='y':
        file=open('eeee.txt','w')
    else:
        pass
else:
    file.write('ssss')
    file.close()