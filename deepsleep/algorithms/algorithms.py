import numpy as np
# from sklearn.preprocessing import Binarizer


class FeatureAlgorithms:

    def findBin(self, X, si):
        # 二值化
        Xs = X.copy()
        Xmean = np.mean(Xs)
        Xbig = []
        Xsmall = []
        for j, Xj in enumerate(Xs):
            if Xj >= Xmean:
                Xbig = Xbig + [j]  # 保存下标
            else:
                Xsmall = Xsmall + [j]
        si = si + 1

        return {'x0': Xsmall, 'x1': Xbig, 'slice_i': si}

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
        global xflagold, xflag, xold
        if coarseLevel <= 2 or coarseLevel % 2 != 0:  # 不是偶数
            raise ValueError('invalid argument: coarseLevel')

        Xs = np.reshape(X.copy(), (1, np.size(X)))

        Xbin = np.zeros(np.shape(Xs),dtype=int)
        slice = int(coarseLevel / 2)
        # Xbig = np.zeros((slice, np.size(Xs)))
        # Xsmall = np.zeros((slice, np.size(Xs)))
        # Xs = np.r_[Xs, Xbig]
        slice_i = 0
        # for slice_i in range(0, slice):  # 0 1
        # x0, x1 = self.findBin(Xs[0])
        # si = 0
        # binSeries = {'x0': Xsmall, 'x1': Xbig, 'slice_i': slice_i}
        binSeries = self.findBin(Xs[0], slice_i)
        binSeries_temp0 = self.findBin(Xs[0][binSeries['x0']], binSeries['slice_i'])
        binSeries_temp1 = self.findBin(Xs[0][binSeries['x1']], binSeries['slice_i'])

        s00 = np.array(binSeries_temp0['x0'])
        s01 = np.array(binSeries_temp0['x1'])
        s10 = np.array(binSeries_temp1['x0'])
        s11 = np.array(binSeries_temp1['x1'])

        s0 = np.array(binSeries['x0'])
        s1 = np.array(binSeries['x1'])

        x0 = s0[s00]
        x1 = s0[s01]
        x2 = s1[s10]
        x3 = s1[s11]

        X0 = Xs[0][x0]
        X1 = Xs[0][x1]
        X2 = Xs[0][x2]
        X3 = Xs[0][x3]
        X0mean = np.mean(X0)
        X1mean = np.mean(X1)
        X2mean = np.mean(X2)
        X3mean = np.mean(X3)
        Xmean = np.mean(Xs)

        # step2 : 将序列的第一个值与平均值比较，第二个值开始二值化结果取决于与前一个点比较
        for i, x in enumerate(Xs[0]):
            if x in X0:
                xflag = 0
            elif x in X1:
                xflag = 1
            elif x in X2:
                xflag = 2
            elif x in X3:
                xflag = 3

            if i >= 1:
                if x < xold and (xflag != xflagold):
                    Xbin[0][i] = 0
                elif x >= xold and (xflag != xflagold ):
                    Xbin[0][i] = 1
                else:
                    Xbin[0][i] = Xbin[0][i-1]
                xold = x
                xflagold = xflag
            elif i == 0 :
                if x < Xmean:
                    Xbin[0][i] = 0
                elif x >= Xmean:
                    Xbin[0][i] = 1
                xold = x
                xflagold = xflag

        return Xbin
