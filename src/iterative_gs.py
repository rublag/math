import numpy as np
import scipy.stats as ss
import wavelets

from typing import Callable, Union

class g_estimator:
    k = 10
    kernel_norm = 1/(2*(k-1))

    xmin = 0.05
    xmax = 100
    xsteps = 2000
    xs = None

    zmin = 0.05
    zmax = 100
    zsteps = 2000
    zs = None

    K = None

    def __init__(self, additive_shift, c=None):
        if c:
            self.xmin = c['xmin']
            self.xmax = c['xmax']
            self.xsteps = c['xsteps']
            self.zmin = c['smin']
            self.smax = c['smax']
            self.ssteps = c['ssteps']
        self.additive_shift = additive_shift
        self.xs = np.linspace(self.xmin, self.xmax, self.xsteps)
        self.zs = np.linspace(self.zmin, self.zmax, self.zsteps)
        
        xzs = self.zs[None, :] / self.xs[:, None]
        self.K = ss.chi2.pdf(xzs, df=2*self.k) / self.xs[:, None]

    def set_kernel(self, k: Callable[[int, int], int], kernel_norm=1):
        """Set kernel for fredholm equation
        k is 
        """
        pass
    
    def get_new_f(self, gs):
        fs = self.K * (gs[None, :] * self.zstep)
        fs = np.sum(fs, axis=1)
        return fs

    def get_new_g(self, g_prev, f_prev, f_true):
        fps = f_prev
        fs = f_true
        gps = g_prev

        gs = self.K * (fs / fps * self.xstep)[:, None]
        gs = gps * np.sum(gs, axis=0) / self.kernel_norm
        return gs

    def get_first_g(self):
        return np.ones(len(self.zs))
    
    def compute(self, f, max_iters=100, max_res=0.05, shift_initial_guess = False):
        fs = f(self.xs)
        fs = fs + self.additive_shift
        assert (fs > 0).all()

        gs = self.get_first_g()
        if shift_initial_guess:
            gs = gs + self.additive_shift

        res = None

        for i in range(max_iters):
            fps = self.get_new_f(gs)

            res = np.max(np.abs(fs - fps))
            res2 = np.mean((fs-fps)**2)
            if res < max_res:
                break

            gs = self.get_new_g(gs, fps, fs)
            print(i, res, res2)
        
        fps = fps - self.additive_shift
        if shift_initial_guess:
            gs = gs - self.additive_shift
        
        return fps, gs, res
