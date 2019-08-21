import numpy as np


class SearchFit(object):
    """Class used for fitting a model multiple times searching across a range of input parameters.
    
    This is a brute-force approach to finding a better fit solution in cases where the standard minimization tools
    are likely to become stuck in a local minimum.
    
    Parameters
    ----------
    model : lmfit Model instance
    parameter_space : list
        Structure that describes the space of fit parameters to search. Each item is a list representing one
        _dimension_ of the parameter space, and the items in a dimension are dictionaries used to modify
        the *params* argument to model.fit().
    kwds : keyword arguments
        Default keyword arguments to use when calling model.fit(). These may be overridden by items in 
        *parameter_space*.

    Examples
    --------
    To fit a gaussian model using a few different initial values for xoffset and amp::
    
        # make some noise with a bump
        y = np.random.normal(size=1000)
        y[220:250] += 10
    
        # try fitting a gaussian with multiple starting values for xoffset
        model = Gaussian()
        amp = [{'amp': -1}, {'amp': 1}]
        xoffset = [{'xoffset':(x, x-50, x+50)} for x in [50, 150, 250, 350, 450]]
        # Total number of fit attempts is 2*5=10
        
        search = SearchFit(model, [amp, xoffset], params={'sigma': (50, 1, 500), 'yoffset': 0}, data=y)
        best = search.best_result
    
    """
    def __init__(self, model, parameter_space, **kwds):
        self.model = model
        self.parameter_space = parameter_space
        self.kwds = kwds
        self.kwds.setdefault('params', {})

        # list of indices to iterate over complete parameter space
        slices = tuple([slice(0, len(p)) for p in self.parameter_space])
        all_inds = np.mgrid[slices]
        self.all_inds = all_inds.reshape(all_inds.shape[0], np.product(all_inds.shape[1:])).T
        
        self.results = None
        self._best_index = None
        
    @property
    def best_result(self):
        if self.results is None:
            for result in self.iter_fit():
                pass
            
        if self._best_index is None:
            all_nrmse = [result['result'].nrmse() for result in self.results]
            self._best_index = np.argmin(all_nrmse)
            
        return self.results[self._best_index]['result']

    def iter_fit(self):
        """Generator that yields results from fitting each point in the parameter space.
        """
        self.results = []
        assert len(self.all_inds) > 0, "No parameters to search"
        for i,inds in enumerate(self.all_inds):
            params = {}
            for j,ind in enumerate(inds):
                params.update(self.parameter_space[j][ind])
            result = self.fit_one(params)
            result = {'param_index': inds, 'params': params, 'result': result}
            self.results.append(result)
            yield result
        
    def fit_one(self, params):
        kwds = self.kwds.copy()
        kwds['params'] = kwds['params'].copy()
        kwds['params'].update(params)
        result = self.model.fit(**kwds)
        return result
        
    def __len__(self):
        return len(self.all_inds)