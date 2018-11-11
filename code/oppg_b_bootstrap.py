from project1.resampling import Bootstrap
from datageneration import *
from numpy import logspace

L = 40
lmb = .001
B = 10
N = 1000
X, y = generate_data(L, N)

for n in range(100, N, 100):
    print("n= %s" % n)
    #Bootstrap(X, y, B, 'ols') # OLS
    #Bootstrap(X, y, B, 'ridge', lmb) # Ridge
    Bootstrap(X[:n,:], y[:n,:], B, 'lasso', lmb) # Lasso

