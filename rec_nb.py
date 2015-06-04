import numpy as np

def load_data():
    data = np.loadtxt('data/letter.p1', dtype='int', delimiter=',')
    print data[10:15]

def start():
    load_data()

if __name__ == '__main__':
    start()