from deepsleep.algorithms import algorithms
import numpy as np
import random
# X = np.array([1,0,1,0,1,-6,1,0,0.6])
# X = np.array([1, 0,0,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0])
X = np.random.random(3200)*100

sLz = algorithms.FeatureAlgorithms()
LZv = sLz.permuteLZ(X)

# import nolds
# randata= np.random.random(10000) # 1000个随机点
# rwalk = np.cumsum(randata)  #累积和
# h = nolds.corr_dim(rwalk,5)