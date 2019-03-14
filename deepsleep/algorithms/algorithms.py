import numpy as np
# from sklearn.preprocessing import Binarizer
# from scipy.special import perm
from itertools import permutations


class FeatureAlgorithms:

    @staticmethod
    def _meanslice(X, si):
        """
        功能: 利用平均值进行划分
        参数：   X：带划分的数据序列
                si：粗粒化水平
        输出： 划分出的两个区域的下标，序列被划分的次数
        """
        # 二值化
        Xs = X.copy()  # 没有二维化
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

    @staticmethod
    def simpleLZ(X):
        """
        功能: 最简单Lemple-Ziv复杂度计算
        参数：   X：原始数据序列

        输出： 复杂度的值
        """
        Xs = np.reshape(X.copy(), (1, np.size(X)))  # 二维化
        Xmean = np.mean(Xs)
        Xbin = np.zeros(np.shape(Xs), dtype=int)  # 二值化序列初始化
        for j, Xj in enumerate(Xs[0]):
            if Xj >= Xmean:
                Xbin[0][j] = 1
            else:
                Xbin[0][j] = 0

        # 初始化子串
        c = 1
        S = str(Xbin[0][0])
        Q = ''
        # 找到新子串的数目
        for i, x in enumerate(Xbin[0]):
            if i >= 1:
                Q = Q + str(x)
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1
        #
        b = len(Xs[0]) / np.log2(len(Xs[0]))
        lzc = c / b
        return lzc

    def coarseLZ(self, X, coarseLevel):
        """
        功能: 粗粒化Lemple-Ziv复杂度计算
        参数：   X：原始数据序列
                coarseLevel：粗粒化水平
        输出： 复杂度的值
        """
        global xflagold, xflag, xold
        if coarseLevel <= 2 or coarseLevel % 2 != 0:  # 不是偶数
            raise ValueError('invalid argument: coarseLevel')
        Xs = np.reshape(X.copy(), (1, np.size(X)))
        Xbin = np.zeros(np.shape(Xs), dtype=int)
        # slice = int(coarseLevel / 2)
        # Xbig = np.zeros((slice, np.size(Xs)))
        # Xsmall = np.zeros((slice, np.size(Xs)))
        # Xs = np.r_[Xs, Xbig]
        slice_i = 0

        # step1： 通过平均值划分几个区域
        binSeries = self._meanslice(Xs[0], slice_i)
        binSeries_temp0 = self._meanslice(Xs[0][binSeries['x0']], binSeries['slice_i'])
        binSeries_temp1 = self._meanslice(Xs[0][binSeries['x1']], binSeries['slice_i'])

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
        # X0mean = np.mean(X0)
        # X1mean = np.mean(X1)
        # X2mean = np.mean(X2)
        # X3mean = np.mean(X3)
        Xmean = np.mean(Xs)

        # step2 : 二值化序列， 将序列的第一个值与平均值比较，第二个值开始二值化结果取决于与前一个点比较
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
                elif x >= xold and (xflag != xflagold):
                    Xbin[0][i] = 1
                else:
                    Xbin[0][i] = Xbin[0][i - 1]

            elif i == 0:
                if x < Xmean:
                    Xbin[0][i] = 0
                elif x >= Xmean:
                    Xbin[0][i] = 1
            xold = x
            xflagold = xflag

        # step3:  对二值序列进行复杂度计算
        # 初始化子串
        c = 1
        S = str(Xbin[0][0])
        Q = ''
        # 找到新子串的数目
        for i, x in enumerate(Xbin[0]):
            if i >= 1:
                Q = Q + str(x)
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1

        b = len(Xs[0]) / np.log2(len(Xs[0]))
        lzc = c / b

        return lzc

    @staticmethod
    def diffLZ(X):
        """
        功能: 差分Lemple-Ziv复杂度计算
        参数：   X：原始数据序列

        输出： 复杂度的值
        """
        global xold
        Xs = np.reshape(X.copy(), (1, np.size(X)))  # 二维化
        Xbin = np.zeros(np.shape(Xs), dtype=int)  # 二值化序列初始化
        # step1： 全序列差分绝对值，再平均，
        D = abs(Xs[0][:-1] - Xs[0][1:])
        T = np.sum(D) / len(D)

        # step2: 二值化序列
        Xmean = np.mean(Xs)
        for i, x in enumerate(Xs[0]):
            if i >= 1:
                if D[i - 1] < T:
                    Xbin[0][i] = Xbin[0][i - 1]
                elif D[i - 1] >= T:
                    if x < xold:
                        Xbin[0][i] = 0
                    elif x >= xold:
                        Xbin[0][i] = 1
            else:
                if x < Xmean:
                    Xbin[0][i] = 0
                else:
                    Xbin[0][i] = 1
            xold = x

        # step3:  对二值序列进行复杂度计算
        # 初始化子串
        c = 1
        S = str(Xbin[0][0])
        Q = ''
        # 找到新子串的数目
        for i, x in enumerate(Xbin[0]):
            if i >= 1:
                Q = Q + str(x)
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1

        b = len(Xs[0]) / np.log2(len(Xs[0]))
        lzc = c / b
        return lzc

    @staticmethod
    def multiLZ(X, window):
        """
        功能: 多尺度Lemple-Ziv复杂度计算
        参数：   X：原始数据序列
                window：滑窗长度
        输出： 复杂度的值
        """
        global xold
        if window <= 4 or window % 2 == 0:  # 是偶数
            raise ValueError('invalid argument: window')
        Xs = np.reshape(X.copy(), (1, np.size(X)))  # 二维化
        Xbin = np.zeros(np.shape(Xs), dtype=int)  # 二值化序列初始化
        Tdw = np.zeros(np.shape(Xs), dtype=float)
        # step1： 滑窗中点
        for i, x in enumerate(Xs[0]):
            if (window - 1) / 2 <= i <= len(Xs[0]) - 1 - ((window - 1) / 2):
                Tdw[0][i] = np.median(Xs[0][i - 2: i + 3])
            else:
                Tdw[0][i] = Xs[0][i]

        # step2: 二值化序列
        for i, x in enumerate(Xs[0]):
            if x < Tdw[0][i]:
                Xbin[0][i] = 0
            else:
                Xbin[0][i] = 1

        # step3:  对二值序列进行复杂度计算
        # 初始化子串
        c = 1
        S = str(Xbin[0][0])
        Q = ''
        # 找到新子串的数目
        for i, x in enumerate(Xbin[0]):
            if i >= 1:
                Q = Q + str(x)
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1

        b = len(Xs[0]) / np.log2(len(Xs[0]))
        lzc = c / b
        return lzc

    @staticmethod
    def permuteLZ(X, t, m):
        """
        功能: 排列Lemple-Ziv复杂度计算
        参数：   X：原始数据序列
                t： 样本点的迟滞因子
                m： 排列模式中包含的样本数目,又叫“嵌入维数”
        输出： 复杂度的值
        """
        if t < 1 and m < 3:  # 参数异常
            raise ValueError('invalid argument: t  and  m')
        Xs = np.reshape(X.copy(), (1, np.size(X)))  # 二维化
        Xsymbol = np.zeros(np.shape(Xs), dtype=int)  # 符号化序列初始化 （符号：1 2 3 4 5 6）

        # step1: 将信号转换成一个排列序列,  （m!）种排列模式，
        lenXs = len(Xs[0])  # 原始数据序列的长度
        permlist = list(permutations(range(0, m), m))  # 生成排列模式
        # motif = np.zeros((len(permlist),), dtype=int)

        Xsend = range(lenXs)
        for j in Xsend:
            if j >= (t * m): # 前面几个值不考虑
                # 重构序列是细粒度平滑移动（重叠）
                Xsvec = Xs[0][j - t * m: j: t]
                # Xmotif = np.sort(Xsvec)
                Xmotif = np.argsort(Xsvec)
                for jj in range(len(permlist)):
                    model = np.array(permlist[jj])
                    if np.all(np.equal(model, Xmotif)):
                        # 重构序列是细粒度平滑移动（重叠）
                        Xsymbol[0][j] = jj
                        break
        # step2:  对符号序列进行复杂度计算
        # 初始化子串
        c = 1
        S = str(Xsymbol[0][0])
        Q = ''
        # 找到新子串的数目
        for i, x in enumerate(Xsymbol[0]):
            if i >= 1:
                Q = Q + str(x)
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1

        b = len(Xs[0]) / np.log2(len(Xs[0]))
        lzc = c / b
        return lzc

    @staticmethod
    def permuteEntropy(X, t, m):
        """
        功能: 排列熵计算
        参数：   X：原始数据序列
                t： 样本点的迟滞因子
                m： 排列模式中包含的样本数目
        """
        if t < 1 and m < 3:  # 参数异常
            raise ValueError('invalid argument: t  and  m')
        Xs = np.reshape(X.copy(), (1, np.size(X)))  # 二维化

        # step1: 将信号转换成一个排列序列,  （m!）种排列模式，
        lenXs = len(Xs[0])
        permlist = list(permutations(range(0, m), m))
        motif = np.zeros((len(permlist),), dtype=int)

        Xend = range(lenXs)
        for j in Xend:
            if j >= (t * m):
                Xsvec = Xs[0][j - t * m: j:t]
                # Xmotif = np.sort(Xsvec)
                Xmotif = np.argsort(Xsvec)
                for jj in range(len(permlist)):
                    model = np.array(permlist[jj])
                    if np.all(np.equal(model, Xmotif)):
                        motif[jj] = motif[jj] + 1
                        break
        # hist = motif.copy()
        # 计算熵值
        motif = motif[(motif != 0)]
        p = motif / sum(motif)
        pe = -sum(p * np.log(p))
        return pe
