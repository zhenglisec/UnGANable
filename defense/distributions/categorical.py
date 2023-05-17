"""
    Code by Tae-Hwan Hung(@graykode)
    https://en.wikipedia.org/wiki/Categorical_distribution
    3-Class Example
"""
import random
import numpy as np
from matplotlib import pyplot as plt

def categorical(p, k):
    return p[k]
def mycategorical(lens):
    n_experiment = lens
    ###
    b = np.random.dirichlet(np.ones(lens), size=1)
    b.sort()
    b = np.around(b, 4).tolist()
    b = b[0]
    p = b
    ###
    x = np.arange(n_experiment)
    y = []
    for _ in range(n_experiment):
        pick = categorical(p, k=random.randint(0, len(p) - 1))
        y.append(pick)
    return y
'''
u, s = np.mean(y), np.std(y)
plt.scatter(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))
plt.legend()
plt.savefig('graph/categorical.png')
plt.show()
'''
