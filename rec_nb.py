import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

def load_data():
    data = np.loadtxt('data/letter.p2', dtype='int', delimiter=',')
    data_train = data
    y = data_train[:,0]
    x = data_train[:, range(1,len(data_train[0]))]
    
    pca = PCA(n_components=2)
    t_data_lst = pca.fit(x).transform(x)
    s_data = np.sort(t_data_lst, axis=0)
    plt.plot(s_data[:,0], s_data[:,1])
    plt.show()
    
#     print t_data_lst
#     print '0 : ',t_data_lst[:,0]
#     print '1 : ',t_data_lst[:,1]
#     x_red = []
#     y_red = []
#     
#     for y_data, t_data in zip(y, t_data_lst):
#         if y_data == -1:
#             plt.plot(t_data[0], t_data[1], color='red')
#         else:
#             plt.plot(t_data[0], t_data[1], color='green')
#     plt.show()
#     ml = GaussianNB()
#     ml.fit(x, y)
#     print 'x', x
#     print 'y', y
#     print ml.predict(x[0]), y[0]
    
def binary_feature():
    data_train = np.loadtxt('data/SPECT.train', dtype='int', delimiter=',')
#     print data_train
    y_train = data_train[:,0]
    x_train = data_train[:, range(1, len(data_train[0]))]
    
    data_test = np.loadtxt('data/SPECT.test', dtype='int', delimiter=',')
    y_test = data_test[:,0]
    x_test = data_test[:, range(1, len(data_test[0]))]
    
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
    print y_train
    ml = GaussianNB()
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    print 'acc : ',accuracy_score(y_test, y_pred)

def start():
    binary_feature()

if __name__ == '__main__':
    start()