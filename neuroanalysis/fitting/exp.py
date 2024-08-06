import warnings
from typing import Callable

import functools
import numpy as np
import scipy.optimize
from scipy.optimize import minimize

from .fitmodel import FitModel
from ..data import TSeries


def exp_decay(t, yoffset, yscale, tau, xoffset=0):
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
    start_y = data.data[:len(data.data)//10].mean()
    end_y = data.data[-len(data.data)//100:].mean()
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


def exp_fit(data: TSeries):
    initial_guess = estimate_exp_params(data)
    # offset, scale, tau
    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
    fn = functools.partial(exp_decay, xoffset=initial_guess[3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = scipy.optimize.curve_fit(
            f=fn,
            xdata=data.time_values,
            ydata=data.data,
            p0=initial_guess[:3],
            bounds=bounds,
            # ftol=1e-8, gtol=1e-8,
        )
    nrmse = normalized_rmse(data, fit[0], fn)
    model = lambda t: fn(t, *fit[0])
    return {
        'fit': fit[0], 
        'result': fit, 
        'nrmse': nrmse,
        'initial_guess': initial_guess,
        'model': model,
    }


def double_exp_fit(data: TSeries, pulse_start: float):
    prepulse_median = np.median(data.time_slice(pulse_start - 5e-3, pulse_start).data)

    def fn(t, pip_yoffset, pip_xoffset, pip_tau, cell_offset, cell_scale, cell_tau):
        amp = prepulse_median - pip_yoffset
        return (exp_decay(t, pip_yoffset, amp, pip_tau, pip_xoffset)
                + exp_decay(t, cell_offset, cell_scale, cell_tau, data.t0 + 150e-6)
                - pip_yoffset)

    initial_guess = estimate_exp_params(data)[:3]
    initial_guess = (
        exp_decay(data.t0, *initial_guess, data.t0 + 150e-6),
        10e-6,
        initial_guess[2] / 10,
        *initial_guess,
    )
    bounds = ([-np.inf, -np.inf, 0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = scipy.optimize.curve_fit(
            f=fn,
            xdata=data.time_values,
            ydata=data.data,
            p0=initial_guess,
            bounds=bounds,
        )
    nrmse = normalized_rmse(data, fit[0], fn)
    model = lambda t: fn(t, *fit[0])
    return {
        'fit': fit[0],
        'result': fit,
        'nrmse': nrmse,
        'initial_guess': initial_guess,
        'model': model,
    }


def fit_double_exp_decay(data: TSeries, pulse: TSeries, base_median: float, pulse_start: float, single_exp_model: Callable):
    prepulse_median = np.median(data.time_slice(pulse_start - 5e-3, pulse_start).data)

    def double_exp_decay(t, yoffset, tau, xoffset):
        amp = prepulse_median - yoffset
        return exp_decay(t, yoffset, amp, tau, xoffset) + single_exp_model(t) - yoffset

    y0 = single_exp_model(pulse.t0)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def fit_with_explicit_hessian(data: TSeries, **kwds):
    """Using an explicit Hessian matrix, fit the model to data."""
    model = functools.partial(exp_decay, xoffset=data.t0)
    p0 = estimate_exp_params(data)[:3]

    def gradient(p, t, y):
        yoffset, scale, tau = p
        y_pred = model(t, yoffset, scale, tau)
        dy_dyoffset = -2 * np.sum(y - y_pred)
        t_over_tau = t / tau
        dy_dscale = -2 * np.sum((y - y_pred) * np.exp(-t_over_tau))
        dy_dtau = -2 * np.sum((y - y_pred) * scale * t * np.exp(-t_over_tau) / tau ** 2)
        return np.array([dy_dyoffset, dy_dscale, dy_dtau])

    def hand_checked_hessian(p, t, y):
        yoffset, scale, tau = p
        t_over_tau = t / tau
        e_to_the_minus_t_over_tau = np.exp(-t_over_tau)

        d2_offset2 = 2 * len(t)
        d2_offsetscale = 2 * np.sum(e_to_the_minus_t_over_tau)
        d2_offsettau = 2 * scale * np.sum(t * e_to_the_minus_t_over_tau) / tau**2
        d2_scaleoffset = 2 * np.sum(e_to_the_minus_t_over_tau)
        d2_scale2 = 2 * np.sum(np.exp(-2 * t / tau))
        d2_scaletau = 2 * np.sum((y - yoffset - 2 * scale * e_to_the_minus_t_over_tau) * t * e_to_the_minus_t_over_tau / tau ** 2)
        d2_tauoffset = d2_offsettau
        d2_tauscale = d2_scaletau
        d2_tau2 = -2 * np.sum((y - yoffset - 2 * scale * e_to_the_minus_t_over_tau) * scale * (t ** 2) * (2 * tau + 1) * e_to_the_minus_t_over_tau / (tau ** 4))

        return np.array([
            [d2_offset2, d2_offsetscale, d2_offsettau],
            [d2_scaleoffset, d2_scale2, d2_scaletau],
            [d2_tauoffset, d2_tauscale, d2_tau2],
        ])

    def hessian(p, t, y):
        yoffset, scale, tau = p
        y_pred = model(t, yoffset, scale, tau)
        d2y_dyoffset2 = 2 * np.sum(1)
        d2y_dyoffsetdscale = 2 * np.sum(np.exp(-t / tau))
        d2y_dyoffsetdtau = 2 * np.sum(scale * t * np.exp(-t / tau) / tau)
        d2y_dscale2 = 2 * np.sum(np.exp(-2 * t / tau))
        d2y_dtau2 = -2 * np.sum((y - y_pred) * scale * t ** 2 * np.exp(-t / tau) / tau ** 4)
        d2y_dscaledtau = -2 * np.sum(t * np.exp(-2 * t / tau) / tau)  # chatgpt
        # d2y_dscaledtau = 2 * np.sum((y - y_pred) * t * np.exp(-t / tau) / tau ** 2)  # copilot
        return np.array([
            [d2y_dyoffset2, d2y_dyoffsetdscale, d2y_dyoffsetdtau],
            [d2y_dyoffsetdscale, d2y_dscale2, d2y_dscaledtau],
            [d2y_dyoffsetdtau, d2y_dscaledtau, d2y_dtau2],
        ])

    def error_function(p, t, y):
        yoffset, scale, tau = p
        return np.sum((y - model(t, yoffset, scale, tau)) ** 2) / len(t)

    result = minimize(
        fun=error_function,
        x0=p0,
        args=(data.time_values, data.data),
        jac=gradient,
        hess=hand_checked_hessian,
        method='trust-ncg',
        # method='trust-exact',
        # method='trust-krylov',
        # method='trust-constr',
        # method='dogleg',
        # method='Newton-CG',
        **kwds,
    )
    return {
        'fit': result.x,
        'result': result,
        'nrmse': normalized_rmse(data, result.x, model),
        'model': lambda t: model(t, *result.x),
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
