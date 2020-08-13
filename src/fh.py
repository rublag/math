import numpy as np
import scipy.optimize
import scipy.linalg
import pickle as pkl
import math
import torch

from numba import jit, njit

# core algorithm of non-negative Tikhonov regularization with equality constraint (NNETR)
@jit
def NNETR(K, f, Delta, epsilon, alpha):
    # the first step
    A_nn = np.vstack((K, alpha * np.identity(K.shape[1])))
    b_nn = np.hstack((f, np.zeros(K.shape[1])))

    # Use NNLS solver provided by scipy
    # res = scipy.optimize.lsq_linear(A_nn, b_nn, verbose=2)['x']
    # res = scipy.linalg.lstsq(A_nn, b_nn)
    at = torch.tensor(A_nn).cuda()
    bt = torch.tensor(b_nn).cuda().unsqueeze(1)
    res = torch.lstsq(bt, at)[0].cpu().numpy().squeeze()[:A_nn.shape[1]]

    # solution should be divided by Delta (grid size)
    # sol = sol/Delta
    return res

@njit
def genfa(f, m, n, low, high, steps):
    xs = np.linspace(low, high, steps)
    return f(xs, m, n)

@njit
def genka(kf, xlow, xhigh, xsteps, slow, shigh, ssteps):
    res = np.empty((xsteps, ssteps))
    print(44)
    for i, x in enumerate(np.linspace(xlow, xhigh, xsteps)):
        for j, s in enumerate(np.linspace(slow, shigh, ssteps)):
            res[i, j] = kf(x, s)
        print('genka x ', x)
    return res

@njit
def psi(t):
    return 2./np.sqrt(3.*np.sqrt(np.pi)) * (1.-t**2) * np.exp(-(t**2)/2)

@njit
def f00(t):
    return psi(t)

@njit
def fmn(t, m, n):
    return 1./np.sqrt((2.)**m) * psi(t/((2.)**m) - n)

@njit
def kf(x, s):
    k = 10
    fac10 = 10*9*8*7*6*5*4*3*2
    fac9  =    9*8*7*6*5*4*3*2
    if x == 0 or s == 0:
        return 0

    return 1/2**k * 1/x * 1/fac9 * (s/x)**(k-1) * np.exp(-s/(2*x))

xmin, xmax, xsteps = 0, 1000, 10000
smin, smax, ssteps = 0, 1000, 10000

ka = genka(kf, xmin, xmax,  xsteps, smin, smax, ssteps)

X = np.random.normal(1, 1/4, 1000)
Y = np.random.chisquare(10, 1000)
Z = X*Y

if __name__ == '__main__':
    all_gs = []
    egs = []
    efs = []
    for m in range(-5, 5):
        m_gs = []
        egs_m = []
        efs_m = []
        for n in range(-5, 5):
            fa = genfa(fmn, m, n, xmin, xmax, xsteps)
            res = NNETR(ka, fa, 0.1, 0.1, 0.1) / 0.1
            m_gs.append(res)
            egs_m.append(np.mean(np.interp(Z, np.linspace(xmin, xmax, xsteps), res)))
            efs_m.append(np.mean(fmn(X, m, n)))
            print(f'm: {m}, n: {n}')
        all_gs.append(m_gs)
        egs.append(egs_m)
        efs.append(efs_m)

    pkl.dump(all_gs, open('res.pkl', 'wb'))
    pkl.dump(egs, open('egs.pkl', 'wb'))
    pkl.dump(efs, open('efs.pkl', 'wb'))

    print('f')
    print(efs)
    print()
    print('g')
    print(egs)
