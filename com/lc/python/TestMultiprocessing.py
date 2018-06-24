import multiprocessing as mp
import threading as td
# window不可以用fork()，但是可以用multiprocessing，可以跨平台使用


def job(a,d):
    print('aaaaa')

def test():
    print("hello")

if __name__ == '__main__':
    t1 = td.Thread(target=job,args=(1,2))
    p1 = mp.Process(target=job,args=(1,2))
    t1.start()
    p1.start()

if __name__ == '__main__':
    # freeze__support()
    p = mp.Process(target=test)
    p.start()

