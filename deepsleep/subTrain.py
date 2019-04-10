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
Xgen = np.zeros([0,0])
ygen = np.zeros([0,0])

# 选取数据源
for u in [2,3,4,6,7,10,11,12,13]:
    X, y = readcsvData(u, X, y, setname='CAP')
# for u in [1, 4, 5, 3, 7, 8, 9, 11, 12]:  #加载XN数据集
#     X, y = readcsvData(u, X, y, setname='XN')
for u in [1,5,8,9]:  #加载CAP数据集  泛化测试集
    Xgen, ygen = readcsvData(u, Xgen, ygen, setname='CAP')
# for u in [10,2]:  #加载XN数据集  泛化测试集
#     Xgen, ygen = readcsvData(u, Xgen, ygen, setname='CAP')

# 构造SVM分类器
singlemodel = SVC(kernel='rbf',  class_weight='balanced')
#超参数网格范围，逐一尝试
param_grid = {'C': np.arange(40, 43), 'gamma': np.arange(0.1,2,0.1)}
start=time.time()
singlemodelGrid = GridSearchCV(singlemodel, param_grid, cv=5, n_jobs=13)  #找到最好的超参数
singlemodelGrid.fit(X, y)  # 输入训练集

end=time.time()
total_time = end - start
print("总耗时:"+str(total_time))

# 预测泛化
ygen_pred = singlemodelGrid.predict(Xgen)
# 打印分类器报告 和 时间
print(metrics.classification_report(ygen_pred, ygen))



# file = open('singleXN_Model.pickle', 'wb')
# pickle.dump(singlemodelGrid, file)
# file.close()

