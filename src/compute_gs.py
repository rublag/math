import json
import math
import numpy as np
import pickle as pkl
import scipy.optimize
import scipy.linalg
import sys
import torch
from tqdm import tqdm

import c_modules.wavelets as wavelets

def lsq_fit_with_regularization(K, f, alpha, device_name='cuda'):
    I = np.identity(K.shape[1]) * alpha
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

def function_discretize(f, m, n, low, high, steps):
    res = np.empty(steps)
    xs = np.linspace(low, high, steps)
    for i, x in enumerate(xs):
        res[i] = f(x, m, n)
    return res

def kernel_discretize(K, xlow, xhigh, xsteps, slow, shigh, ssteps):
    res = np.empty((xsteps, ssteps))
    for i, x in tqdm(enumerate(np.linspace(xlow, xhigh, xsteps))):
        for j, s in enumerate(np.linspace(slow, shigh, ssteps)):
            res[i, j] = K(x, s)
    return res

#@njit
psi = wavelets.meyer

#@njit
f_mn = wavelets.f_mn(psi)

def fac(x):
    res = 1
    for i in range(1, x+1):
        res *= i
    return res

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
    'smax': 1000,
    'ssteps': 10_000,

    'mmin': -5,
    'mmax': 5,

    'nmin': -5,
    'nmax': 5,

    'save': 'res.pkl',
    'device': 'cuda',

    'alpha': 0.1
}

USAGE = "python fh.py [config.json]"

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

    dx = (xmax - xmin) / (xsteps - 1)
    ds = (smax - smin) / (ssteps - 1)

    print(f"Computing for m: [{mmin}, {mmax}], n: [{nmin}, {nmax}]")

    gs_all = []
    errs_all = []
    for m in range(mmin, mmax+1):
        gs_fixed_m = []
        errs_m = []
        for n in range(nmin, nmax+1):
            print(f"Discretizing f_{{{m}, {n}}}")
            f_discretized = function_discretize(f_mn, m, n, xmin, xmax, xsteps)
            print("Computing g")
            # Discarding residual for now
            gds, errs = lsq_fit_with_regularization(K_discretized, f_discretized, 0.1, device_name)
            gs_fixed_m.append(gds / ds)
            errs_m.append(errs)
            print(f"Finished with m: {m}, n: {n}")
        gs_all.append(gs_fixed_m)
        errs_all.append(errs_m)

    result = {
        'gs': gs_all,
        'config': config,
        #'K_f': K_f,
        'K_discretized': K_discretized,
        'dx': dx,
        'ds': ds,
        'err': errs_all
        #'f_mn': f_mn,
        #'psi': psi
    }
    pkl.dump(result, open(savefile, 'wb'))
