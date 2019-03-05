import numpy as np
from sklearn.preprocessing import Binarizer
class FeatureAlgorithms:
    def simpleLZ(self, X):
        Xs = X.copy()
        # if LZv <= 1:
        #     raise ValueError('invalid argument')
        # 二值化
        Xmean = np.mean(Xs)
        Xbin = Xs
        for j, Xj in enumerate(Xs) :
            if Xj >= Xmean:
                Xbin[j] = 1
            else:
                Xbin[j] = 0
        # 转换成int型
        Xbin = np.asanyarray(Xbin,dtype=int)
        # 初始化子串
        c = 1
        S = str(Xbin[0])
        Q = ''
        SQ = ''
        # 找到新子串的数目
        for i, x in enumerate(Xbin):
            if i >= 1:
                Q = Q + str(x)
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1
        #
        b = len(Xs) / np.log2(len(Xs))
        lzc = c / b
        return lzc

    def multiscaleLZ(self, X, coarseLevel):  # 粗粒化水平
        if coarseLevel >= 2:
            raise ValueError('invalid argument')

        return




