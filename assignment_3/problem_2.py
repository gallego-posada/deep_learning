import numpy as np

for n in [1, 10, 100, 500, 1000, 100000000]:
    p = np.random.rand(n)
    naive_logp = np.log(np.prod(p))
    stable_logp = np.sum(np.log(p))
    print(n, naive_logp, stable_logp)
