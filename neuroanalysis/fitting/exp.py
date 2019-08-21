from __future__ import print_function, division

import numpy as np
from .fitmodel import FitModel


class Exp(FitModel):
    """Single exponential fitting model.
    
    Parameters are xoffset, yoffset, amp, and tau.
    """
    def __init__(self):
        FitModel.__init__(self, self.exp, independent_vars=['x'], nan_policy='omit', method='nelder')

    @staticmethod
    def exp(x, xoffset, yoffset, amp, tau):
        return yoffset + amp * np.exp(-(x - xoffset)/tau)

    def fit(self, *args, **kwds):
        kwds.setdefault('method', 'nelder')
        return FitModel.fit(self, *args, **kwds)


class Exp2(FitModel):
    """Double exponential fitting model.
    
    Parameters are xoffset, yoffset, amp, tau1, and tau2.
    
        exp2 = yoffset + amp * (exp(-(x-xoffset) / tau1) - exp(-(x-xoffset) / tau2))

    """
    def __init__(self):
        FitModel.__init__(self, self.exp2, independent_vars=['x'])

    @staticmethod
    def exp2(x, xoffset, yoffset, amp, tau1, tau2):
        xoff = x - xoffset
        out = yoffset + amp * (np.exp(-xoff/tau1) - np.exp(-xoff/tau2))
        out[xoff < 0] = yoffset
        return out

