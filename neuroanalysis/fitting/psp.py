from __future__ import print_function, division

import numpy as np
import scipy.optimize
from .fitmodel import FitModel


class Psp(FitModel):
    """PSP-like fitting model defined as the product of rising and decaying exponentials.
    
    Parameters
    ----------
    x : array or scalar
        Time values
    xoffset : scalar
        Horizontal shift (positive shifts to the right)
    yoffset : scalar
        Vertical offset
    rise_time : scalar
        Time from beginning of psp until peak
    decay_tau : scalar
        Decay time constant
    amp : scalar
        The peak value of the psp
    rise_power : scalar
        Exponent for the rising phase; larger values result in a slower activation
    
    Notes
    -----
    This model is mathematically similar to the double exponential used in
    Exp2 (the only difference being the rising power). However, the parameters
    are re-expressed to give more direct control over the rise time and peak value.
    This provides a flatter error surface to fit against, avoiding some of the
    tradeoff between parameters that Exp2 suffers from.
    """
    # default guess / bounds:

    # if guess is None:
    #     guess = [
    #         (y.max()-y.min()) * 2,
    #         0, 
    #         x[-1]*0.25,
    #         x[-1]
    #     ]
    
    # if bounds is None:
    #     bounds = [[None,None]] * 4
    #     bounds[1][0] = -2e-3
    #     minTau = (x[1]-x[0]) * 0.5
    #     #bounds[2] = [minTau, None]
    #     #bounds[3] = [minTau, None]

    def __init__(self):
        FitModel.__init__(self, self.psp_func, independent_vars=['x'])

    @staticmethod
    def _psp_inner(x, rise, decay, power):
        return (1.0 - np.exp(-x / rise))**power * np.exp(-x / decay)

    @staticmethod
    def _psp_max_time(rise, decay, rise_power):
        """Return the time from start to peak for a psp with given parameters."""
        return rise * np.log(1 + (decay * rise_power / rise))

    @staticmethod
    def psp_func(x, xoffset, yoffset, rise_time, decay_tau, amp, rise_power):
        """Function approximating a PSP shape. 

        """
        rise_tau = Psp._compute_rise_tau(rise_time, rise_power, decay_tau)
        #decay_tau = (rise_tau / rise_power) * (np.exp(rise_time / rise_tau) - 1)
        max_val = Psp._psp_inner(rise_time, rise_tau, decay_tau, rise_power)
        
        xoff = x - xoffset
        output = np.empty(xoff.shape, xoff.dtype)
        output[:] = yoffset
        mask = xoff >= 0
        output[mask] = yoffset + (amp / max_val) * Psp._psp_inner(xoff[mask], rise_tau, decay_tau, rise_power)
        
        if not np.all(np.isfinite(output)):
            raise ValueError("Parameters are invalid: xoffset=%f, yoffset=%f, rise_tau=%f, decay_tau=%f, amp=%f, rise_power=%f, isfinite(x)=%s" % (xoffset, yoffset, rise_tau, decay_tau, amp, rise_power, np.all(np.isfinite(x))))
        return output
            
    @staticmethod
    def _compute_rise_tau(rise_time, rise_power, decay_tau):
        fn = lambda tr: tr * np.log(1 + (decay_tau * rise_power / tr)) - rise_time
        return scipy.optimize.fsolve(fn, (rise_time,))[0]


class StackedPsp(FitModel):
    """A PSP on top of an exponential decay.
    
    Parameters are the same as for Psp, with the addition of *exp_amp*, which
    is the amplitude of the underlying exponential decay at the onset of the
    psp.
    """
    def __init__(self):
        FitModel.__init__(self, self.stacked_psp_func, independent_vars=['x'])
    
    @staticmethod
    def stacked_psp_func(x, xoffset, yoffset, rise_time, decay_tau, amp, rise_power, exp_amp):
        exp = exp_amp * np.exp(-(x-xoffset) / decay_tau)
        return exp + Psp.psp_func(x, xoffset, yoffset, rise_time, decay_tau, amp, rise_power)


class PspTrain(FitModel):
    """A Train of PSPs, all having the same rise/decay kinetics.
    """
    def __init__(self, n_psp):
        self.n_psp = n_psp
        def fn(*args, **kwds):
            return self.psp_train_func(n_psp, *args, **kwds)
        
        # fn.argnames and fn.kwargs are used internally by lmfit to override
        # its automatic argument detection
        fn.argnames = ['x', 'xoffset', 'yoffset', 'rise_time', 'decay_tau', 'rise_power']
        fn.kwargs = []
        for i in range(n_psp):
            fn.argnames.extend(['xoffset%d'%i, 'amp%d'%i])
            fn.kwargs.append(('decay_tau_factor%d'%i, None))
        
        FitModel.__init__(self, fn, independent_vars=['x'])

    @staticmethod
    def psp_train_func(n_psp, x, xoffset, yoffset, rise_time, decay_tau, rise_power, **kwds):
        """Paramters are the same as for the single Psp model, with the exception
        that the x offsets and amplitudes of each event must be numbered like
        xoffset0, amp0, xoffset1, amp1, etc.
        """
        for i in range(n_psp):
            xoffi = kwds['xoffset%d'%i]
            amp = kwds['amp%d'%i]
            tauf = kwds.get('decay_tau_factor%d'%i, 1)
            psp = Psp.psp_func(x, xoffset+xoffi, 0, rise_time, decay_tau*tauf, amp, rise_power)
            if i == 0:
                tot = psp
            else:
                tot += psp
        
        return tot + yoffset



class Psp2(FitModel):
    """PSP-like fitting model with double-exponential decay.
    
    Shape is computed as the product of a rising exponential and the sum of two decaying exponentials.

    Parameters are xoffset, yoffset, slope, and amp.
    """
    def __init__(self):
        FitModel.__init__(self, self.double_psp_func, independent_vars=['x'])

    @staticmethod
    def double_psp_func(x, xoffset, yoffset, rise_tau, decay_tau1, decay_tau2, amp1, amp2, rise_power=2.0):
        """Function approximating a PSP shape with double exponential decay. 
        """
        x = x-xoffset
        
        out = np.zeros(x.shape, x.dtype)
        mask = x >= 0
        x = x[mask]
        
        rise_exp = (1.0 - np.exp(-x / rise_tau))**rise_power
        decay_exp1 = amp1 * np.exp(-x / decay_tau1)
        decay_exp2 = amp2 * np.exp(-x / decay_tau2)
        out[mask] = riseExp * (decay_exp1 + decay_exp2)

        return out


def create_all_fit_param_combos(base_params):
    '''Convert the parameters fed to fit_psp into a list of all possible parameter 
    dictionaries to be fed to PSP() or stackedPSP() for fitting. 
    
    Parameters 
    ----------
    base_params: dictionary
        Each value in the key:value dictionary pair must be a tuple.
        In general the structure of the tuple is of the form, 
        (initial conditions, lower boundary, higher boundary).
        The initial conditions can be either a number or a list 
        of numbers specifying several initial conditions.  The 
        initial condition may also be fixed by replacing the lower 
        higher boundary combination with 'fixed'.
        Note that technically the value of a key:value pair could 
        be a single string (this ends up being evaluated as an 
        expression by lmfit later on).
    
    Returns
    -------
    param_dict_list: list of dictionaries
        Each dictionary contains parameter inputs for one fit_psp run.
        
    Examples:    
    base_params[amplitude]=(10, 0, 20)
    base_params[amplitude]=(10, 'fixed')
    base_params[amplitude]=([5,10, 20], 0, 20)
    base_params[amplitude]=([5,10, 20], 'fixed')
    
    '''
    # need to create all combinations of the initial conditions
    param_dict_list = [{}] #initialize list
    for key, value in base_params.items():
        if isinstance(value[0], list):
            temp_list=[]
            for init_cond in value[0]: #for each initial condition
                temp=[pdl.copy() for pdl in param_dict_list] #copy each parameter dictionary in the list so they do not point back to the original dictionary
                for t in temp:  #for each dictionary in the list 
                    t[key]=tuple([init_cond] +list(value[1:])) #add the key and single initial condition pair
                temp_list=temp_list+temp
            param_dict_list=list(temp_list) #Note this works because the dictionaries are already made immutable above
        else: 
            for pdl in param_dict_list:
                pdl[key]=value
    
    return param_dict_list


def fit_psp(response, 
            xoffset, # this parameter will be fit.
            clamp_mode='ic', 
            sign='any', #Note this will not be used if *amp* input is specified
            method='leastsq', 
            fit_kws=None, 
            stacked=True,
            rise_time_mult_factor=10., #Note this will not be used if *rise_time* input is specified 
            weight=None,
            amp_ratio=None, 
            # the following are parameters that can be fit 
            amp=None,
            decay_tau=None,
            rise_power=None,
            rise_time=None,
            yoffset=None
            ):
    """Fit psp waveform to the equation specified in the PSP class in neuroanalysis.fitting

    Parameters
    ----------
    response : neuroanalysis.data.TSeries class
        Contains data on trace waveform.
    clamp_mode : string
        either 'ic' for current clamp or 'vc' for voltage clamp
    sign : string
        Specifies the sign of the PSP deflection.  Must be '+', '-', or any. If *amp* 
        is specified, value will be irrelevant.
    method : string 
        Method lmfit uses for optimization
    rise_time_mult_factor: float
        Parameter that goes into the default calculation rise time.  
        Note that if an input for *rise_time* is provided this input
        will be irrelevant.
    stacked : True or False
        If True, use the :class:`StackedPsp` model function. This model fits
        the PSP shape on top of a baseline exponential decay, which is useful when the
        PSP follows close after another PSP or action potential.
        See *amp_ratio* to bound the amplitude of the baseline exponential.
    weight : numpy array, size equal to data waveform, default: np.ones(len(response.data))
        assigns relative weights to each data point in waveform for fitting.
    fit_kws : dict
        Additional key words that are fed to lmfit
    The parameters below are fed to the psp function. Each value in the 
        key:value dictionary pair must be a tuple.
        In general the structure of the tuple is of the form, 
        (initial conditions, lower boundary, higher boundary).
        The initial conditions can be either a number or a list 
        of numbers specifying several initial conditions.  The 
        initial condition may also be fixed by replacing the lower 
        higher boundary combination with 'fixed'. If nothing is 
        supplied defaults will be used.    
        Examples:    
            amplitude=(10, 0, 20)
            amplitude=(10, 'fixed')
            amplitude=([5,10, 20], 0, 20)
            amplitude=([5,10, 20], 'fixed') 
        xoffset : tuple
            Time where psp begins in reference to the start of *response*.
            Note that this parameter must be specified by user.
            Example: ``(14e-3, -float('inf'), float('int'))``
        yoffset : tuple
            Vertical offset of rest.  Note that default of zero assumes rest has been subtracted from traces.
            Default is ``(0, -float('inf'), float('inf')``.
        rise_time : tuple
            Time from beginning of psp until peak. Default initial condition is 5 ms for current clamp
            or 1 ms for voltage clamp. Default bounds are calculated using *rise_time_mult_factor*.
        decay_tau : tuple
            Decay time constant. Default initial condition is 50 ms for current clamp
            or 4 ms for voltage clamp. Default bounds are from 0.1 to 10 times the initial value.
        rise_power : tuple
            Exponent for the rising phase; larger values result in a slower activation. Default is
            ``(2.0, 'fixed')``.
        amp_ratio : tuple
            Ratio of the amplitude of the baseline exponential decay to the amplitude of the PSP.
            This parameter is used when *stacked* is True in order to bound the amplitude of the
            baseline exponential.
    
    Returns
    -------
    fit : lmfit.model.ModelResult
        Best fit
    """           
    
    # extracting these for ease of use
    t = response.time_values
    y = response.data
    dt = response.dt
    
    # set initial conditions depending on whether in voltage or current clamp
    # note that sign of these will automatically be set later on based on the 
    # the *sign* input
    if clamp_mode == 'ic':
        amp_init = .2e-3
        amp_max = 100e-3
        rise_time_init = 5e-3
        decay_tau_init = 50e-3
    elif clamp_mode == 'vc':
        amp_init = 20e-12
        amp_max = 500e-12
        rise_time_init = 1e-3
        decay_tau_init = 4e-3
    else:
        raise ValueError('clamp_mode must be "ic" or "vc"')

    # Set up amplitude initial values and boundaries depending on whether *sign* are positive or negative
    if sign == '-':
        amp_default = (-amp_init, -amp_max, 0)
    elif sign == '+':
        amp_default = (amp_init, 0, amp_max)
    elif sign == 'any':
        warnings.warn("You are not specifying the predicted sign of your psp.  This may slow down or mess up fitting")
        amp_default = (0, -amp_max, amp_max)
    else:
        raise ValueError('sign must be "+", "-", or "any"')
        
    # initial condition, lower boundry, upper boundry    
    base_params = {
        'xoffset': xoffset,
        'yoffset': yoffset or (0, -float('inf'), float('inf')),
        'rise_time': rise_time or (rise_time_init, rise_time_init/rise_time_mult_factor, rise_time_init*rise_time_mult_factor),
        'decay_tau': decay_tau or (decay_tau_init, decay_tau_init/10., decay_tau_init*10.),
        'rise_power': rise_power or (2, 'fixed'),
        'amp': amp or amp_default,
    }
    
    # specify fitting function and set up conditions
    if not isinstance(stacked, bool):
        raise Exception("Stacked must be True or False")
    if stacked:
        psp = StackedPsp()
        base_params.update({
            #TODO: figure out the bounds on these
            'amp_ratio': amp_ratio or (0, -100, 100),
            'exp_amp': 'amp * amp_ratio',
        })  
    else:
        psp = Psp()
    
    # set weighting that 
    if weight is None: #use default weighting
        weight = np.ones(len(y))
    else:  #works if there is a value specified in weight
        if len(weight) != len(y):
            raise Exception('the weight and array vectors are not the same length') 
    
    # arguement to be passed through to fitting function
    fit_kws = {'weights': weight}   

    # convert initial parameters into a list of dictionaries to be consumed by psp.fit()        
    param_dict_list = create_all_fit_param_combos(base_params)

    # cycle though different parameters sets and chose best one
    best_fit = None
    best_score = None
    for p in param_dict_list:
        fit = psp.fit(y, x=t, params=p, fit_kws=fit_kws, method=method)
        err = np.sum(fit.residual**2)  # note: using this because normalized (nrmse) is not necessary to comparing fits within the same data set
        if best_fit is None or err < best_score:
            best_fit = fit
            best_score = err
    fit = best_fit

    # nrmse = fit.nrmse()
    if 'baseline_std' in response.meta:
        fit.snr = abs(fit.best_values['amp']) / response.meta['baseline_std']
        fit.err = fit.nrmse() / response.meta['baseline_std']

    return fit

