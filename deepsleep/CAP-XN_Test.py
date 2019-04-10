"""
用CAP和XN 相互测试

"""
import pandas as pd
import time
from sklearn.svm import SVC  # "Support vector classifier"
from sklearn.model_selection import train_test_split, GridSearchCV  # 划分训练和测试集
import re
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle

from sklearn.model_selection import learning_curve


def readcsvData(u, X, y, setname):
    if setname == 'CAP':
        dataFile = "../CAP-features/CAP%03d_featureSSS.csv" % (u)
    elif setname == 'XN':
        dataFile = "../XN-features/XN%03d_featureSSS.csv" % (u)
    else:
        return None
    data_df = pd.read_csv(dataFile)
    data_set = data_df.values
    if X.any():
        X = np.append(X, data_set[:, :30], 0)
        y = np.append(y, data_set[:, 30], 0)
    else:
        X = data_set[:, :30]
        y = data_set[:, 30]
    return X, y


X = np.zeros([0, 0])
y = np.zeros([0, 0])
Xtest = np.zeros([0, 0])
ytest = np.zeros([0, 0])
# 选取数据源
for u in [1,2,3,3,5,6,7,8,9,10,11,12,12]:
    X, y = readcsvData(u, X, y, setname='CAP')
for u in [1,2,3,4,5,6,7,8,9,10,11,12]:
    X, y = readcsvData(u, X, y, setname='XN')

for u in [4]:
    Xtest, ytest = readcsvData(u, Xtest, ytest, setname='CAP')

# 构造SVM分类器
bigmodel = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
# bigmodel = SVC(kernel='rbf', gamma='auto')
# 划分训练集和测试集
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=14)
# 超参数网格范围，逐一尝试
# param_grid = {'C': np.arange(5, 50), 'gamma': np.arange(0.01,2,0.003)}
start = time.time()

# 加载之前训练好的模型
# file = open('save_results/singleXN_Model.pickle', 'rb')
# bigmodelGrid = pickle.load(file)
# file.close()
# bigmodelGrid = GridSearchCV(bigmodel, param_grid, cv=5, n_jobs=13)  #找到最好的超参数
# bigmodelGrid.fit(Xtrain, ytrain)
bigmodel.fit(X[:, :20], y)
# 预测
ypred = bigmodel.predict(Xtest[:, :20])

# 打印分类器报告 和 时间
print(metrics.classification_report(ypred, ytest))
end = time.time()
total_time = end - start
print("总耗时:" + str(total_time))

"""
绘制学习曲线
"""
# train_sizes, train_scores, valid_scores = learning_curve(bigmodel, X[:,6:10], y, train_sizes=np.arange(0.1,1,0.01), cv=5, n_jobs=13)
#
# import matplotlib.pyplot as plt
# new_train_scores = train_scores.mean(1)
# train_std = train_scores.std()
# test_std = valid_scores.std()
# new_test_scores = valid_scores.mean(1)
#
# plt.grid()
# plt.fill_between(train_sizes, new_train_scores - train_std,
#                  new_train_scores + train_std, color='r', alpha=0.1)
# plt.fill_between(train_sizes, new_test_scores - test_std,
#                  new_test_scores + test_std, color='g', alpha=0.1)
# plt.plot(train_sizes, new_train_scores, '*-', c='r', label='train score')
# plt.plot(train_sizes, new_test_scores, '*-', c='g', label='test score')
# plt.legend(loc='best')
# plt.show()
