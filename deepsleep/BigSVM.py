"""
用CAP和XN所有人训练集进行SVM训练一个大模型

"""
import pandas as pd
import time
from sklearn.svm import SVC    # "Support vector classifier"
from sklearn.model_selection import train_test_split, GridSearchCV  #划分训练和测试集
import re
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pickle
def readcsvData(u, X, y, setname):
    if setname == 'CAP':
        dataFile = "CAP-features/CAP%03d_featureSSS.csv" % (u)
    elif setname == 'XN':
        dataFile = "XN-features/XN%03d_featureSSS.csv" % (u)
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


X = np.zeros([0,0])
y = np.zeros([0,0])
# for u in range(1, 14):
#     X, y = readcsvData(u, X, y, setname='CAP')
for u in range(1, 13):  #加载XN数据集
    X, y = readcsvData(u, X, y, setname='XN')
bigmodel = SVC(kernel='rbf',  class_weight='balanced')
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=13)
param_grid = {'C': np.arange(5, 50), 'gamma': np.arange(0.01,2,0.003)}  #超参数网格范围，逐一尝试
start=time.time()

file = open('bigmodelGrid.pickle', 'rb')
bigmodelGrid = pickle.load(file)
file.close()
# bigmodelGrid = GridSearchCV(bigmodel, param_grid, cv=5, n_jobs=13)  #找到最好的超参数
# bigmodelGrid.fit(Xtrain, ytrain)
ypred = bigmodelGrid.predict(X)

# 打印分类器报告 和 时间
print(metrics.classification_report(ypred, y))
end=time.time()
total_time = end - start
print("总耗时:"+str(total_time))

