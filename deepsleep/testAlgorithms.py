from deepsleep.algorithms import algorithms
import numpy as np
import random
# X = np.array([1,0,1,0,1,-6,1,0,0.6])
# X = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0])
X = np.random.random(100)

sLz = algorithms.FeatureAlgorithms()
LZv = sLz.multiscaleLZ(X,4)

# if LZv <= 1:
#     raise ValueError('invalid argument')

# for j in X:
#     print(j[0])
# from sklearn.preprocessing import Binarizer
#
# biner = Binarizer(threshold = np.mean(X) )
# X_bin = biner.transform(X)

# X.reshape(-1, 1)
# biner = Binarizer()
# X = biner.transform(X)