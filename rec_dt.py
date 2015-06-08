import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

def rec_decsion_tree():
    data = np.loadtxt('data/letter.p2', dtype=int, delimiter=',')
    y = data[:, 0]
    x = data[:, range(1, len(data[0]))]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
    ml = GaussianNB()
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    print accuracy_score(y_test, y_pred)
        
#     x_select = SelectKBest(chi2, k=3).fit_transform(x,y)
#     remove_lst = [6,10,13]
    x = np.delete(x, 6 ,1)
    x = np.delete(x, 10 ,1)
    x = np.delete(x, 13 ,1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
    ml = GaussianNB()
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    print accuracy_score(y_test, y_pred)
        
#     print x[0]
#     print x_select
#     row_select_lst = []
#     print x
#     for i in range(len(x[0])):
#         row = x[:,i]
# #         print row
#         for row_select in x_select:
#             if np.array_equal(row, row_select) :
#                 row_select_lst.append(i)
#     print row_select_lst
#     print x[0], x[1]
#     print x_new
#     print chi2(x, y)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)
#     ml = DecisionTreeClassifier()
#     ml.fit(x_train, y_train)
#     y_pred = ml.predict(x_test)
#     print accuracy_score(y_test, y_pred)
    

def start():
    rec_decsion_tree()

if __name__ == '__main__':
    start()