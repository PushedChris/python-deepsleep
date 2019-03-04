import numpy as np




class FeatureAlgorithms:

    def simpleLZ(self, X):
        c = 1
        S = X[0]
        Q = ''
        SQ = ''
        for i, x in enumerate(X):
            if i >= 1:
                Q = Q + x
                SQ = S + Q
                SQv = SQ[0: -1]
                if Q not in SQv:
                    S = SQ
                    Q = ''
                    c = c + 1
        b = len(X) / np.log2(len(X))
        lzc = c / b
        return lzc


# X = '10101010'
# sLz = FeatureAlgorithm()
# sLz.simpleLZ(X)


