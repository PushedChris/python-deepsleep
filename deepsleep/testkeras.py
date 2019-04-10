import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

import pandas as pd
# import time
# from sklearn.svm import SVC    # "Support vector classifier"
# from sklearn.model_selection import train_test_split, GridSearchCV  #划分训练和测试集
# import re
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn import preprocessing
# import pickle

from sklearn.model_selection import learning_curve


def readcsvData(u, X, y, setname):
    if setname == 'CAP':
        dataFile = "../CAP-features/CAP%03d_featureSSS.csv" % (u)
    elif setname == 'XN':
        dataFile = "../XN-features/XN%03d_featureSSS.csv" % (u)
    else:
        return None
    data_df = pd.read_csv(dataFile)
    data_set  = data_df.values
    if X.any():
        X = np.append(X, data_set[:, :30], 0)
        y = np.append(y, data_set[:, 30], 0)
    else:
        X = data_set[:, :30]
        y = data_set[:, 30]
    return X, y


x_train = np.zeros([0,0])
y_train = np.zeros([0,0])
x_test = np.zeros([0,0])
y_test = np.zeros([0,0])

# 选取数据源
for u in [3,4,5]:
    x_train, y_train = readcsvData(u, x_train, y_train, setname='XN')
for u in [7]:
    x_test, y_test = readcsvData(u, x_test, y_test, setname='XN')

# 生成虚拟数据
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))
# L = np.size(x_train, 0)
model = Sequential()
model.add(Dense(81, input_dim=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(81, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(81, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train , y_train)
score = model.evaluate(x_test, y_test)
print(score)

train_sizes, train_scores, valid_scores = learning_curve(model, x_train, y_train, train_sizes=range(1,250,1), cv=6)