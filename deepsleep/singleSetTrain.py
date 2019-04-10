"""
单个样本集进行SVM训练一个模型

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


X = np.zeros([0,0])
y = np.zeros([0,0])

# 选取数据源
for u in range(1, 14):
    X, y = readcsvData(u, X, y, setname='CAP')
# for u in range(1, 13):  #加载XN数据集
#     X, y = readcsvData(u, X, y, setname='XN')

# 构造SVM分类器
singlemodel = SVC(kernel='rbf',  class_weight='balanced')
# 划分训练集和测试集
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.9, random_state=10)
#超参数网格范围，逐一尝试
param_grid = {'C': np.arange(1, 10), 'gamma': np.arange(0.1,2,0.1)}
start=time.time()

singlemodelGrid = GridSearchCV(singlemodel, param_grid, cv=5, n_jobs=13)  #找到最好的超参数
singlemodelGrid.fit(Xtrain, ytrain)  # 输入训练集

# 预测
ypred = singlemodelGrid.predict(Xtest)

# 打印分类器报告 和 时间
print(metrics.classification_report(ypred, ytest))
end=time.time()
total_time = end - start
print("总耗时:"+str(total_time))

# file = open('singleXN_Model.pickle', 'wb')
# pickle.dump(singlemodelGrid, file)
# file.close()

