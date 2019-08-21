from __future__ import print_function, division

import numpy as np
from .fitmodel import FitModel


class Sigmoid(FitModel):
    """Sigmoid fitting model.
    
    Parameters are xoffset, yoffset, slope, and amp.
    """
    def __init__(self):
        FitModel.__init__(self, self.sigmoid, independent_vars=['x'])

    @staticmethod
    def sigmoid(x, xoffset, yoffset, slope, amp):
        return amp / (1.0 + np.exp(-slope * (x-xoffset))) + yoffset

