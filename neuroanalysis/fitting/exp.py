import functools
import numpy as np
import scipy.optimize
from .fitmodel import FitModel


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


def normalized_rmse(data, params):
    y = exp_decay(data.time_values, *params)
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
