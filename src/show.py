import numpy as np
import pickle as pkl
import scipy.stats as ss
import sys

from matplotlib import pyplot as plt
from numba import jit, njit


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: show.py result_filename")
        sys.exit(1)

    res = pkl.load(open(sys.argv[1], 'rb'))

    psi = res['psi']
    f_mn = res['f_mn']
    gs_all = res['gs']
    config = res['config']

    X = np.random.chisquare(5, 1000)
    Y = np.random.chisquare(20, 1000)
    Z = X*Y

    xmin, xmax, xsteps = config['xmin'], config['xmax'], config['xsteps']
    smin, smax, ssteps = config['smin'], config['smax'], config['ssteps']
    mmin, mmax = config['mmin'], config['mmax']
    nmin, nmax = config['nmin'], config['nmax']
    savefile = config['save']
    device_name = config['device']

    egs = []
    efs = []
    for m in range(mmin, mmax+1):
        egs_m = []
        efs_m = []
        for n in range(nmin, nmax+1):
            egs_m.append(np.mean(np.interp(Z, np.linspace(xmin, xmax, xsteps), gs_all[m-mmin][n-nmin], right=0)))
            efs_m.append(np.mean(f_mn(X, m, n)))
        egs.append(egs_m)
        efs.append(efs_m)


    ts = np.arange(0, 10, 0.001)
    fy = gy = ry = 0

    c = 2/(3.223+3.596)


    for m in range(mmin, mmax+1):
        for n in range(nmin, nmax+1):
            fy = fy + c*efs[m-mmin][n-nmin]*f_mn(ts, m, n)
            gy = gy + c*egs[m-mmin][n-nmin]*f_mn(ts, m, n)

    #ry = ss.norm.pdf(ts, 1, 1/4)
    ry = ss.chi2.pdf(ts, 5)
    #ry = ss.expon.pdf(ts)

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 5)
    ax.plot(ts, fy, label='wavelet reconstruction ($ \\sum E \\psi_{mn} (X))$)')
    ax.plot(ts, gy, label='estimation ($\\sum E g_{mn} (Z)$)')
    ax.plot(ts, ry, label='real density')
    ax.legend()
    plt.show()
