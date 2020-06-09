import pickle as pkl
import numpy as np
import scipy.stats as ss

from matplotlib import pyplot as plt

def psi(t):
    return 2./np.sqrt(3.*np.sqrt(np.pi)) * (1.-t**2) * np.exp(-(t**2)/2)

def fmn(t, m, n):
    return 1./np.sqrt((2.)**m) * psi(t/((2.)**m) - n)

#X = np.random.normal(1, 1/4, 1000)
X = np.random.chisquare(5, 100000)
#X = np.random.exponential(1, 100000)
Y = np.random.chisquare(20, 100000)
Z = X*Y
print(max(Z))

xmin, xmax, xsteps = 0, 100, 1000

egs = []
efs = []
all_gs = pkl.load(open('res.pkl', 'rb'))
for m in range(-5, 5):
    egs_m = []
    efs_m = []
    for n in range(-5, 5):
        egs_m.append(np.mean(np.interp(Z, np.linspace(xmin, xmax, xsteps), all_gs[m+5][n+5], right=0)))
        efs_m.append(np.mean(fmn(X, m, n)))
        print(f'm: {m}, n: {n}')
    egs.append(egs_m)
    efs.append(efs_m)


ts = np.arange(0, 10, 0.001)
fy = gy = ry = 0

c = 2/(3.223+3.596)


for m in range(-5, 5):
    for n in range(0, 5):
        fy = fy + c*efs[m+5][n+5]*fmn(ts, m, n)
        gy = gy + c*egs[m+5][n+5]*fmn(ts, m, n)

#ry = ss.norm.pdf(ts, 1, 1/4)
ry = ss.chi2.pdf(ts, 5)
#ry = ss.expon.pdf(ts)

plt.plot(ts, fy, label='wavelet reconstruction ($ \\sum E \\psi_{mn} (X))$)')
plt.plot(ts, gy, label='estimation ($\\sum E g_{mn} (Z)$)')
plt.plot(ts, ry, label='real density')
plt.legend()
plt.show()

