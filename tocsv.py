"""

解析mat数据
保存CSV文件
"""
import pandas as pd
import scipy.io as scio  # 读写数据文件
import numpy as np
from sklearn import preprocessing
import csv

dataFile = 'CAP-features/CAP006_featureSS.mat'
matdata = scio.loadmat(dataFile)  # 加载mat文件
data_feature = matdata['data_feature']
data_featureSum=np.zeros((0,0))
for EEGch in range(0, 5):
    if data_feature[0, EEGch].any():  # 非空导联
        ystr = np.transpose(data_feature[1, EEGch][:, 2])
        y = np.zeros(ystr.shape, dtype=int)
        # 将字符串型标签映射成数字标签
        for i, i_ystr in enumerate(list(ystr)):
            strpat = ''.join(i_ystr[0])
            if strpat == 'mix':
                y[i] = 1
            else:
                y[i] = 2
        if data_featureSum.any():   #导联均值化
            data_featureSum = data_featureSum + data_feature[0, EEGch]
        else:
            data_featureSum = np.zeros(np.shape(data_feature[0, EEGch]))
            data_featureSum = data_featureSum + data_feature[0, EEGch]
# min-max标准化(归一化)
min_max_scaler = preprocessing.MinMaxScaler()
Xdf = pd.DataFrame(min_max_scaler.fit_transform(data_featureSum))
ydf = pd.DataFrame(y, columns = ['stages'])
DataSet = pd.concat([Xdf,ydf], axis=1)
DataSet.to_csv('CAP-features/CAP006_featureSSS.csv', index=False)  # SSS ：三个 smooth single


d = 1 + 1
