import json
import math
import numpy as np
import pickle as pkl
import scipy.optimize
import scipy.linalg
import sys
import torch

from numba import jit, njit

@jit(forceobj=True)
def lsq_fit_with_regularization(K, f, alpha, device_name='cuda'):
    I = np.identity(K.shape[1])
    K_long = np.vstack((K, I))

    z = np.zeros((K.shape[1], 1))
    f_long = np.vstack((f.reshape(-1, 1), z))

    device = torch.device(device_name)
    K_tensor = torch.tensor(K_long).to(device)
    f_tensor = torch.tensor(f_long).to(device)

    res = torch.lstsq(f_tensor, K_tensor)[0].cpu().numpy().squeeze()

    g = res[:K_tensor.shape[1]]
    residual = res[K_tensor.shape[1]:]

    return (g, residual)

@njit
def function_discretize(f, m, n, low, high, steps):
    res = np.empty(steps)
    xs = np.linspace(low, high, steps)
    for i, x in enumerate(xs):
        res[i] = f(x, m, n)
    return res

@njit
def kernel_discretize(K, xlow, xhigh, xsteps, slow, shigh, ssteps):
    res = np.empty((xsteps, ssteps))
    for i, x in enumerate(np.linspace(xlow, xhigh, xsteps)):
        for j, s in enumerate(np.linspace(slow, shigh, ssteps)):
            res[i, j] = K(x, s)
    return res

@njit
def psi(t):
    """Mother wavelet function"""
    return 2./np.sqrt(3.*np.sqrt(np.pi)) * (1.-t**2) * np.exp(-(t**2)/2)

@njit
def f_mn(t, m, n):
    return 1./np.sqrt((2.)**m) * psi(t/((2.)**m) - n)

@njit
def fac(x):
    res = 1
    for i in range(1, x+1):
        res *= i
    return res

@njit
def K_f(x, s):
    k = 10
    if x == 0 or s == 0:
        return 0

    return 1/2**k * 1/x * 1/fac(k-1) * (s/x)**(k-1) * np.exp(-s/(2*x))

DEFAULT_CONFIG = {
    'xmin': 0,
    'xmax': 100,
    'xsteps': 1_000,

    'smin': 0,
    'smax': 100,
    'ssteps': 1_000,

    'mmin': -5,
    'mmax': 5,

    'nmin': -5,
    'nmax': 5,

    'save': 'res.pkl',
    'device': 'cuda',

    'alpha': 0.1
}

USAGE = "python fh.py [config.json]"

xmin, xmax, xsteps = 0, 1000, 10000
smin, smax, ssteps = 0, 1000, 10000



if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Invalid number of arguments")
    elif len(sys.argv) == 2:
        fd = open(sys.argv[1], 'r')
        config = {**DEFAULT_CONFIG, **json.load(fd)}
    else:
        config = DEFAULT_CONFIG

    # Unpack all values from config
    xmin, xmax, xsteps = config['xmin'], config['xmax'], config['xsteps']
    smin, smax, ssteps = config['smin'], config['smax'], config['ssteps']
    mmin, mmax = config['mmin'], config['mmax']
    nmin, nmax = config['nmin'], config['nmax']
    savefile = config['save']
    device_name = config['device']

    print("Discretizing kernel K(x, s)")
    print(f"x: range [{xmin}, {xmax}] step {xsteps}")
    print(f"s: range [{smin}, {smax}] step {ssteps}")
    K_discretized = kernel_discretize(K_f, xmin, xmax,  xsteps, smin, smax, ssteps)

    dx = (smax - smin + 1) / ssteps

    print(f"Computing for m: [{mmin}, {mmax}], n: [{nmin}, {nmax}]")

    gs_all = []
    for m in range(mmin, mmax+1):
        gs_fixed_m = []
        for n in range(nmin, nmax+1):
            print(f"Discretizing f_{{{m}, {n}}}")
            f_discretized = function_discretize(f_mn, m, n, xmin, xmax, xsteps)
            print("Computing g")
            # Discarding residual for now
            gdx, _ = lsq_fit_with_regularization(K_discretized, f_discretized, 0.1, device_name)
            gs_fixed_m.append(gdx / dx)
            print(f"Finished with m: {m}, n: {n}")
        gs_all.append(gs_fixed_m)

    result = {
        'gs': gs_all,
        'config': config,
        'K_f': K_f,
        'f_mn': f_mn,
        'psi': psi
    }
    pkl.dump(result, open(savefile, 'wb'))
