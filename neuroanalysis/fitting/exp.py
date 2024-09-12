import warnings

import numpy as np
from scipy.optimize import minimize
from typing import Callable

from .fit_scale_offset import fit_scale_offset
from .fitmodel import FitModel
from ..data import TSeries


def exp_decay(t, yoffset, yscale, tau, xoffset=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yoffset + yscale * np.exp(-(t-xoffset) / tau)


def estimate_exp_params(data: TSeries):
    """Estimate parameters for an exponential fit to data.

    Parameters
    ----------
    data : TSeries
        Data to fit.

    Returns
    -------
    params : tuple
        (yoffset, yscale, tau, toffset)
    """
    start_y = data.data[:len(data.data)//100].mean()
    end_y = data.data[-len(data.data)//10:].mean()
    yscale = start_y - end_y
    yoffset = end_y
    cs = np.cumsum(data.data - yoffset)
    if yscale > 0:
        tau_i = np.searchsorted(cs, cs[-1] * 0.63)
    else:
        tau_i = len(cs) - np.searchsorted(cs[::-1], cs[-1] * 0.63)
    tau = data.time_values[min(tau_i, len(data)-1)] - data.time_values[0]
    return yoffset, yscale, tau, data.t0


def normalized_rmse(data, params, fn: Callable=exp_decay):
    y = fn(data.time_values, *params)
    return np.mean((y - data.data) ** 2)**0.5 / data.data.std()


def best_exp_fit_for_tau(tau, x, y, std=None):
    """Given a curve defined by x and y, find the yoffset and yscale that best fit 
    an exponential decay with a fixed tau.

    Parameters
    ----------
    tau : float
        Decay time constant.
    x : array
        Time values.
    y : array
        Data values to fit.
    std : float
        Standard deviation of the data. If None, it is calculated from *y*.

    Returns
    -------
    yscale : float
        Y scaling factor for the exponential decay.
    yoffset : float
        Y offset for the exponential decay.
    err : float
        Normalized root mean squared error of the fit.
    exp_y : array
        The exponential decay curve that best fits the data.
    
    """
    if std is None:
        std = y.std()
    exp_y = exp_decay(x, tau=tau, yscale=1, yoffset=0)
    yscale, yoffset = fit_scale_offset(y, exp_y)
    exp_y = exp_y * yscale + yoffset
    err = ((exp_y - y) ** 2).mean()**0.5 / std
    return yscale, yoffset, err, exp_y


def quantify_confidence(tau: float, memory: dict, data: TSeries) -> float:
    """
    Given a run of best_exp_fit_for_tau, quantify the confidence in the fit.
    """
    # errs = np.array([v[2] for v in memory.values()])
    # std = errs.std()
    # n = len(errs)
    # data_range = errs.max() - errs.min()
    # max_std = (data_range / 2) * np.sqrt((n - 1) / n)
    # poor_variation = 1 - std / max_std

    y = data.data
    x = data.time_values
    err = memory[tau][2]
    scale, offset = np.polyfit(x, y, 1)
    linear_y = scale * x + offset
    linear_err = ((linear_y - y) ** 2).mean()**0.5 / y.std()
    exp_like = 1 / (1 + err / linear_err)
    exp_like = max(0, exp_like - 0.5) * 2

    # pv_factor = 1
    # el_factor = 4
    # return ((poor_variation ** pv_factor) * (exp_like ** el_factor)) ** (1 / (pv_factor + el_factor))
    return exp_like


def exp_fit(data: TSeries):
    """Fit *data* to an exponential decay.

    This is a minimization of the normalized RMS error of the fit over the decay time constant.
    Other parameters are determined exactly for each value of the decay time constant.
    """
    xoffset = data.t0
    data = data.copy()
    data.t0 = 0
    tau_init = 0.5 * (data.time_values[-1])
    memory = {}
    std = data.data.std()

    def err_fn(params):
        τ = params[0]
        # keep a record of all tau values visited and their corresponding fits
        if τ not in memory:
            memory[τ] = best_exp_fit_for_tau(τ, data.time_values, data.data, std)
        return memory[τ][2]

    result = minimize(
        err_fn,
        tau_init,
        bounds=[(1e-9, None)],
    )

    tau = float(result.x[0])
    yscale, yoffset, err, exp_y = memory[tau]
    return {
        'fit': (yoffset, yscale, tau),
        'result': result,
        'memory': memory,
        'nrmse': err,
        'confidence': quantify_confidence(tau, memory, data),
        'model': lambda t: exp_decay(t, yoffset, yscale, tau, xoffset),
    }


class Exp(FitModel):
    """Single exponential decay fitting model.
    
    Parameters are xoffset, yoffset, amp, and tau.
    """
    def __init__(self):
        FitModel.__init__(self, self.exp, independent_vars=['x'], nan_policy='omit', method='least-squares')

    @staticmethod
    def exp(x, xoffset, yoffset, amp, tau):
        return exp_decay(x - xoffset, yoffset, amp, tau)
    
    def fit(self, *args, **kwds):
        kwds.setdefault('method', 'nelder')
        return FitModel.fit(self, *args, **kwds)


class ParallelCapAndResist(FitModel):
    @staticmethod
    def current_at_t(t, v_over_parallel_r, v_over_total_r, tau, xoffset=0):
        exp = np.exp(-(t - xoffset) / tau)
        return v_over_total_r * (1 - exp) + v_over_parallel_r * exp

    def __init__(self):
        super().__init__(self.current_at_t, independent_vars=['t'], nan_policy='omit', method='least-squares')


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
