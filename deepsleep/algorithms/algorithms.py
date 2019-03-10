import numpy as np
from sklearn.preprocessing import Binarizer


class FeatureAlgorithms:


    def findBin(self, X, si):
        # 二值化
        Xs = X.copy()
        Xmean = np.mean(Xs)
        Xbig = []
        Xsmall = []
        for j, Xj in enumerate(Xs):
            if Xj >= Xmean:
                Xbig = Xbig+[j] # 保存下标
            else:
                Xsmall = Xsmall+[j]
        si = si+1
        return  {'x0':Xsmall, 'x1':Xbig, 'slice_i':si}

    def simpleLZ(self, X):
        Xs = X.copy()
        # if LZv <= 1:
        #     raise ValueError('invalid argument')
        # 二值化
        Xmean = np.mean(Xs)
        Xbin = Xs
        for j, Xj in enumerate(Xs):
            if Xj >= Xmean:
                Xbin[j] = 1
            else:
                Xbin[j] = 0
        # 转换成int型
        Xbin = np.asanyarray(Xbin, dtype=int)
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

    def multiscaleLZ(self, X, coarseLevel):  # coarseLevel：粗粒化水平
        if coarseLevel <= 2 or coarseLevel % 2 != 0:  # 不是偶数
            raise ValueError('invalid argument: coarseLevel')

        Xs = np.reshape(X.copy(), (1, np.size(X)))
        Xmean = np.mean(Xs)
        Xbin = np.zeros(np.shape(Xs))
        slice = np.asanyarray(coarseLevel / 2, dtype=int)
        Xbig = np.zeros((slice, np.size(Xs)))
        Xsmall = np.zeros((slice, np.size(Xs)))
        Xs = np.r_[Xs, Xbig]
        slice_i=0
        # for slice_i in range(0, slice):  # 0 1
        # x0, x1 = self.findBin(Xs[0])

        x0, x1, slice_i  = self.findBin(Xs[0][x0], slice_i)
        x0, x1 = self.findBin(Xs[0][x1])

        return Xbin
