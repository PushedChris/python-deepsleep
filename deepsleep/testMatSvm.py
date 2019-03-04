"""
测试SVM效果
解析mat数据
"""
import pandas as pd
import scipy.io as scio    #读写数据文件
from sklearn.svm import SVC    # "Support vector classifier"
from sklearn.model_selection import train_test_split, GridSearchCV  #划分训练和测试集
import re
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing



dataFile = 'CAP-features/CAP001_featureSS.mat'
matdata = scio.loadmat(dataFile)   # 加载mat文件
data_feature  = matdata['data_feature']
X = pd.DataFrame(data_feature[0, 3]).values #转换第一导联的特征空间为DF格式数据
ystr = pd.DataFrame(data_feature[1,3][:, 2]).values

y = np.zeros(ystr.shape, dtype=int)

# 将字符串型标签映射成数字标签
for i, i_ystr in enumerate(list(ystr)):
    strpat = ''.join(i_ystr[0])
    if strpat == 'mix':
        y[i] = 1
    else:
        y[i] = 2
model = SVC(kernel='rbf', class_weight='balanced', C=10, gamma='auto')
min_max_scaler = preprocessing.MinMaxScaler()
# 归一化
X = min_max_scaler.fit_transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(X[:,0:19], y, random_state=123)
model.fit(Xtrain, ytrain)
# param_grid = {'svc__C': [0.01, 0.05, 0.1, 0.5], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}  #超参数网格范围，逐一尝试
# grid1 = GridSearchCV(model, param_grid)  #找到最好的超参数
# print(grid1.best_params_)
# model = grid1.best_estimator_ #保存最优的SVM模型
# 预测
ypred = model.predict(Xtest)
# 打印分类器报告
print(metrics.classification_report(ypred, ytest))
