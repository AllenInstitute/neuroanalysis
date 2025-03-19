"""
Derived from acq4 and cnmodel code originally developed by Luke Campagnola and Paul B. Manis,
Univerity of North Carolina at Chapel Hill.
"""
import numpy as np
try:
    import lmfit
    LmfitModel = lmfit.Model
except ImportError as exc:
    class LmfitModel:
        def __init__(self, *args, **kwds):
            raise exc

from ..stats import weighted_std
from ..util.optional_import import optional_import
FitExplorer = optional_import('neuroanalysis.ui.fitting', 'FitExplorer')


class FitModel(LmfitModel):
    """ Simple extension of lmfit.Model that allows one-line fitting.
    
    Each subclass of lmfit.Model describes a single mathematical function to be fit.
    The major benefit of lmfit is that it allows any of the function parameters to
    be fixed or bounded arbitrarily.

    Example uses:
        
        # single exponential fit
        fit = expfitting.Exp1.fit(
            data, 
            x=time_vals,
            params=dict(
                xoffset=(0, 'fixed'),           # x offset is fixed at 0
                yoffset=(yoff_guess, -120, 0),  # y offset is bounded between -120 and 0
                amp=(amp_guess, 0, 50),         # amp is bounded between 0 and 50
                tau=(tau_guess, 0.1, 50),       # tau is bounded between 0.1 and 50
            ))
        # plot the fit
        fit_curve = fit.eval()
        plot(time_vals, fit_curve)
        
    
        # double exponential fit with tau ratio constraint
        # note that 'tau_ratio' does not appear in the exp2 model; 
        # we can define new parameters here.
        fit = expfitting.Exp2.fit(
            data, 
            x=time_vals,
            params=dict(
                xoffset=(0, 'fixed'),
                yoffset=(yoff_guess, -120, 0),
                amp1=(amp_guess, 0, 50),
                tau1=(tau_guess, 0.1, 50),
                amp2=(-0.5, -50, 0),
                tau_ratio=(10, 3, 50),          # tau_ratio is bounded between 3 and 50
                tau2='tau1 * tau_ratio'         # tau2 is forced to be tau1 * tau_ratio 
            ))
        
    """
    def fit(self, data, params=None, interactive=False, **kwds):
        """ Return a fit of data to this model.
        
        Parameters
        ----------
        data : array
            dependent data to fit
        interactive : bool
            If True, show a GUI used for interactively exploring fit parameters
            
        Extra keyword arguments are passed to make_params() if they are model
        parameter names, or passed directly to Model.fit() for independent
        variable names.
        """
        if params is None:
            params = {}
        p = self.make_params(**params)
        fit = lmfit.Model.fit(self, data, params=p, **kwds)
        if interactive:
            self.show_interactive(fit)

        # monkey-patch some extra GOF metrics:
        fit.rmse = lambda: self.rmse(fit)
        fit.nrmse = lambda: self.nrmse(fit)

        return fit
        
    def make_params(self, **params):
        """
        Make parameters used for fitting with this model.
        
        Keyword arguments are used to generate parameters for the fit. Each 
        parameter may be specified by the following formats:
        
        param=value :
            The initial value of the parameter
        param=(value, 'fixed') :
            Fixed value for the parameter
        param=(value, min, max) :
            Initial value and min, max values, which may be float or None
        param='expression' :
            Expression used to compute parameter value. See:
            http://lmfit.github.io/lmfit-py/constraints.html#constraints-chapter
        """
        p = lmfit.Parameters()
        for k in self.param_names:
            p.add(k)
        
        for param,val in params.items():
            if param not in p:
                p.add(param)
                
            if isinstance(val, str):
                p[param].expr = val
            elif np.isscalar(val):
                p[param].value = val
            elif isinstance(val, tuple):
                if len(val) == 2:
                    assert val[1] == 'fixed'
                    p[param].value = val[0]
                    p[param].vary = False
                elif len(val) == 3:
                    p[param].value = val[0]
                    p[param].min = val[1] if val[1] is not None else -float('inf')
                    p[param].max = val[2] if val[2] is not None else float('inf')
                else:
                    raise TypeError("Tuple parameter specifications must be (val, 'fixed')"
                                    " or (val, min, max).")
            else:
                raise TypeError("Invalid parameter specification: %r" % val)
            
        # set initial values for parameters with mathematical constraints
        # this is to allow fit.eval(**fit.init_params)
        global_ns = np.__dict__
        for param,val in params.items():
            if isinstance(val, str):
                p[param].value = eval(val, global_ns, p.valuesdict())
        return p

    def show_interactive(self, fit=None):
        """ Show an interactive GUI for exploring fit parameters.
        """
        from ..ui.fitting import FitExplorer

        if not hasattr(self, '_interactive_win'):
            self._interactive_win = FitExplorer(model=self, fit=fit)
        self._interactive_win.show()

    @staticmethod
    def rmse(result):
        residual = result.residual
        return (residual**2 / residual.size).sum() ** 0.5

    @staticmethod
    def nrmse(result):
        rmse = FitModel.rmse(result)
        if result.weights is None:
            std = result.data.std()
        else:
            std = weighted_std(result.data, result.weights)
        return rmse / std
