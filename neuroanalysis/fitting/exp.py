from typing import Callable

import functools
import numpy as np
import scipy.optimize
from .fitmodel import FitModel
from ..data import TSeries


def exp_decay(t, yoffset, yscale, tau, xoffset=0):
    return yoffset + yscale * np.exp(-(t-xoffset) / tau)


def estimate_exp_params(data):
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


def exp_fit(data):
    initial_guess = estimate_exp_params(data)
    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
    fit = scipy.optimize.curve_fit(
        f=functools.partial(exp_decay, xoffset=initial_guess[3]),
        xdata=data.time_values, 
        ydata=data.data, 
        p0=initial_guess[:3], 
        bounds=bounds, 
        # ftol=1e-8, gtol=1e-8,
    )
    nrmse = normalized_rmse(data, fit[0])
    model = lambda t: exp_decay(t, *fit[0], xoffset=initial_guess[3])
    return {
        'fit': fit[0], 
        'result': fit, 
        'nrmse': nrmse,
        'initial_guess': initial_guess,
        'model': model,
    }


def fit_double_exp_decay(data: TSeries, pulse: TSeries, base_median: float, pulse_start: float, transientless_model: Callable):
    prepulse_median = np.median(data.time_slice(pulse_start - 5e-3, pulse_start).data)

    def double_exp_decay(t, yoffset, tau, xoffset):
        amp = prepulse_median - yoffset
        return exp_decay(t, yoffset, amp, tau, xoffset) + transientless_model(t) - yoffset

    y0 = transientless_model(pulse.t0)
    initial_guess = (
        y0,
        10e-6,
        pulse_start,
    )
    bounds = tuple(zip(
        sorted((y0 + y0 - base_median, base_median)),  # yoffset. y0 ± (y0 - base_median). sorted for clearer math.
        (0, 200e-6),  # tau
        (pulse_start - 5e-6, pulse_start + 100e-6),  # xoffset
    ))
    fit_region = data.time_slice(pulse_start, pulse_start + 5e-3)
    result = scipy.optimize.curve_fit(
        f=double_exp_decay,
        xdata=fit_region.time_values,
        ydata=fit_region.data,
        p0=initial_guess,
        bounds=bounds,
        # ftol=1e-8, gtol=1e-8,
    )
    fit = result[0]
    nrmse = normalized_rmse(pulse, fit, double_exp_decay)
    return {
        'fit': fit,
        'result': result,
        'nrmse': nrmse,
        'initial_guess': initial_guess,
        'model': lambda t: double_exp_decay(t, *fit),
        'guessed_model': lambda t: double_exp_decay(t, *initial_guess),
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
