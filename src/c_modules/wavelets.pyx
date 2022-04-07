import numpy as np
from math import pi, sin, cos, sqrt

cimport numpy as np

def mhat(t):
    """Mother wavelet function"""
    return 2./np.sqrt(3.*np.sqrt(np.pi)) * (1.-t**2) * np.exp(-(t**2)/2)

cdef double _meyer(double t):
    """Mother wavelet function"""
    # Meyer wavelet
    cdef double psi1_numerator, psi1_denominator
    cdef double psi2_numerator, psi2_denominator
    cdef double psi1, psi2

    psi1_numerator = (4./(3.*pi) * (t - 1./2.) * cos(2.*pi/3. * (t - 1./2.)) - 1./pi * sin(4.*pi/3. * (t - 1./2.)))
    psi1_denominator = ((t - 1./2.) - 16./9. * (t-1./2.)**3)

    psi2_numerator = (8./(3.*pi) * (t - 1./2.) * cos(8.*pi/3. * (t - 1./2.)) + 1./pi * sin(4.*pi/3. * (t - 1./2.)))
    psi2_denominator = ((t - 1./2.) - 64./9. * (t-1./2.)**3)
   
    psi1 = psi1_numerator / psi1_denominator
    psi2 = psi2_numerator / psi2_denominator
    
    return psi1 + psi2

cdef double _interp(double a, double b, double x, double y, double t):
    return x * ((t-a)/(b-a)) + y * ((b-t)/(b-a))

cdef double _meyer_nonvectorized(double t):
    cdef double eps, bad_t, bad_val

    eps = 1e-10

    cdef double *bad_ts = [-1./4., 1./2., 5./4., 1./8., 7./8.]
    cdef double *bad_vals = [-(8.+3.*pi)/(9.*pi),
                             4./pi,
                             -(8.+3.*pi)/(9.*pi),
                             4.*(-5.+2.*sqrt(2.))/(9.*pi),
                             4.*(-5.+2.*sqrt(2.))/(9.*pi)]
    
    for i in range(5):
        bad_t = bad_ts[i]
        bad_val = bad_vals[i]

        if abs(bad_t - t) < eps:
            if t < bad_t:
                return _interp(bad_t-eps, bad_t, _meyer(bad_t-eps), bad_val, t)
            elif t > bad_t:
                return _interp(bad_t, bad_t+eps, bad_val, _meyer(bad_t+eps), t)
            else:
                return bad_val

    return _meyer(t)

cdef void _meyer_on_array(np.ndarray[np.float64_t, ndim=1] ts, np.ndarray[np.float64_t, ndim=1] ys, size_t n):
    for i in range(n):
        ys[i] = _meyer_nonvectorized(ts[i])

def fmeyer(ts):
    if np.isscalar(ts):
        return _meyer_nonvectorized(ts)

    ts = np.array(ts, dtype=np.float64)

    if ts.ndim != 1:
        raise

    ys = np.empty(len(ts), dtype=np.float64)
    _meyer_on_array(ts, ys, len(ts))
    return ys

meyer = np.vectorize(_meyer_nonvectorized)

def f_mn(psi):
    def inner(t, m, n):
        return 1./np.sqrt((2.)**m) * np.vectorize(psi)(t/((2.)**m) - n)
    return inner
