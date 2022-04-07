import numpy as np
from c_modules.wavelets import meyer

def mhat(t):
    """Mother wavelet function"""
    return 2./np.sqrt(3.*np.sqrt(np.pi)) * (1.-t**2) * np.exp(-(t**2)/2)

def f_mn(psi):
    def inner(t, m, n):
        return 1./np.sqrt((2.)**m) * psi(t/((2.)**m) - n)
    return inner
