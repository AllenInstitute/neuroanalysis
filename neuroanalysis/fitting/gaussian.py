from __future__ import print_function, division

import numpy as np
from .fitmodel import FitModel


class Gaussian(FitModel):
    """Gaussian fitting model.
    
    Parameters are xoffset, yoffset, sigma, and amp.
    """
    def __init__(self):
        FitModel.__init__(self, self.gaussian, independent_vars=['x'])

    @staticmethod
    def gaussian(x, xoffset, yoffset, sigma, amp):
        return amp * np.exp(-((x-xoffset)**2) / (2 * sigma**2)) + yoffset

