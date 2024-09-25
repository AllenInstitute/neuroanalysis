import numpy as np
import scipy.optimize
from .data import TSeries
from .fitting.psp import Psp


def zero_crossing_events(data, min_length=3, min_peak=0.0, min_sum=0.0, noise_threshold=None):
    """Locate events of any shape in a signal. Works by finding regions of the signal
    that deviate from noise, and filtering with multiple criteria such as peak value,
    area under the event, and duration of the event.
    
    This algorithm relies on having data where the signal is lower frequency than the noise.
    Any low frequency noise must be removed by high-pass filtering to produce a flat baseline.
    In addition, the event amplitude must be sufficiently large that the signal does not 
    cross 0 within a single event (low-pass filtering may be necessary to achive this).
    
    Returns an array of events where each row is (start, length, sum, peak)
    """
    def measureNoise(data, threshold=2.0, iterations=2):
        ## Determine the base level of noise
        data = data.view(np.ndarray)
        if iterations > 1:
            med = np.median(data)
            std = data.std()
            thresh = std * threshold
            arr = np.ma.masked_outside(data, med - thresh, med + thresh)
            return measureNoise(arr[~arr.mask], threshold, iterations-1)
        else:
            return data.std()

    def fit(function, xVals, yVals, guess, errFn=None, measureError=False, generateResult=False, resultXVals=None, **kargs):
        """fit xVals, yVals to the specified function. 
        If generateResult is True, then the fit is used to generate an array of points from function
        with the xVals supplied (useful for plotting the fit results with the original data). 
        The result x values can be explicitly set with resultXVals."""
        if errFn is None:
            errFn = lambda v, x, y: function(v, x)-y
        if len(xVals) < len(guess):
            raise Exception("Too few data points to fit this function. (%d variables, %d points)" % (len(guess), len(xVals)))
        fitResult = scipy.optimize.leastsq(errFn, guess, args=(xVals, yVals), **kargs)
        error = None
        #if measureError:
            #error = errFn(fit[0], xVals, yVals)
        result = None
        if generateResult or measureError:
            if resultXVals is not None:
                xVals = resultXVals
            result = function(fitResult[0], xVals)
            #fn = lambda i: function(fit[0], xVals[i.astype(int)])
            #result = fromfunction(fn, xVals.shape)
            if measureError:
                error = abs(yVals - result).mean()
        return fitResult + (result, error)

    def fitGaussian(xVals, yVals, guess=[1.0, 0.0, 1.0, 0.0], **kargs):
        """Returns least-squares fit parameters for function v[0] * exp(((x-v[1])**2) / (2 * v[2]**2)) + v[3]"""
        return fit(gaussian, xVals, yVals, guess, **kargs)

    def gaussian(v, x):
        """Gaussian function value at x. The parameter v is [amplitude, x-offset, sigma, y-offset]"""
        return v[0] * np.exp(-((x-v[1])**2) / (2 * v[2]**2)) + v[3]


    if isinstance(data, TSeries):
        xvals = data.time_values
        data1 = data.data
    else:
        xvals = None
        data1 = data.view(np.ndarray)
    
    ## find all 0 crossings
    mask = data1 > 0
    diff = mask[1:] - mask[:-1]  ## mask is True every time the trace crosses 0 between i and i+1
    times1 = np.argwhere(diff)[:, 0]  ## index of each point immediately before crossing.
    
    times = np.empty(len(times1)+2, dtype=times1.dtype)  ## add first/last indexes to list of crossing times
    times[0] = 0                                         ## this is a bit suspicious, but we'd rather know
    times[-1] = len(data1)                               ## about large events at the beginning/end
    times[1:-1] = times1                                 ## rather than ignore them.
    #p.mark('find crossings')
    
    ## select only events longer than min_length.
    ## We do this check early for performance--it eliminates the vast majority of events
    long_events = np.argwhere(times[1:] - times[:-1] > min_length)
    if len(long_events) < 1:
        n_events = 0
    else:
        long_events = long_events[:, 0]
        n_events = len(long_events)
    
    ## Measure sum of values within each region between crossings, combine into single array
    if xvals is None:
        events = np.empty(n_events, dtype=[('index',int),('len', int),('sum', float),('peak', float)])  ### rows are [start, length, sum]
    else:
        events = np.empty(n_events, dtype=[('index',int),('time',float),('len', int),('sum', float),('peak', float)])  ### rows are [start, length, sum]
    #p.mark('empty %d -> %d'% (len(times), n_events))
    #n = 0
    for i in range(n_events):
        t1 = times[long_events[i]]+1
        t2 = times[long_events[i]+1]+1
        events[i]['index'] = t1
        events[i]['len'] = t2-t1
        evData = data1[t1:t2]
        events[i]['sum'] = evData.sum()
        if events[i]['sum'] > 0:
            peak = evData.max()
        else:
            peak = evData.min()
        events[i]['peak'] = peak
    #p.mark('generate event array')
    
    if xvals is not None:
        events['time'] = xvals[events['index']]
    
    if noise_threshold > 0:
        ## Fit gaussian to peak in size histogram, use fit sigma as criteria for noise rejection
        stdev = measureNoise(data1)
        #p.mark('measureNoise')
        hist = np.histogram(events['sum'], bins=100)
        #p.mark('histogram')
        histx = 0.5*(hist[1][1:] + hist[1][:-1]) ## get x values from middle of histogram bins
        #p.mark('histx')
        fit = fitGaussian(histx, hist[0], [hist[0].max(), 0, stdev*3, 0])
        #p.mark('fit')
        sigma = fit[0][2]
        minSize = sigma * noise_threshold
        
        ## Generate new set of events, ignoring those with sum < minSize
        #mask = abs(events['sum'] / events['len']) >= minSize
        mask = abs(events['sum']) >= minSize
        #p.mark('mask')
        events = events[mask]
        #p.mark('select')

    if min_peak > 0:
        events = events[abs(events['peak']) > min_peak]
    
    if min_sum > 0:
        events = events[abs(events['sum']) > min_sum]
    
    return events


def threshold_events(trace, threshold, adjust_times=True, baseline=0.0, omit_ends=True):
    """
    Finds regions in a TSeries that cross a threshold value (as measured by distance from baseline). Returns the index, length, peak, and sum of each event.
    Optionally adjusts index to an extrapolated baseline-crossing.
    """
    threshold = abs(threshold)
    data = trace.data
    data1 = data - baseline
    #if (hasattr(data, 'implements') and data.implements('MetaArray')):
    
    ## find all threshold crossings
    hits = []
    for signed_threshold in (-threshold, threshold):
        if len(data1) == 0:
            continue
        
        if signed_threshold < 0:
            mask = data1 < signed_threshold
        else:
            mask = data1 > signed_threshold
        
        imask = mask.astype('byte')
        diff = imask[1:] - imask[:-1]
        on_inds = list(np.argwhere(diff==1)[:,0] + 1)
        off_inds = list(np.argwhere(diff==-1)[:,0] + 1)
        
        if omit_ends:
            if mask[0]:
                off_inds = off_inds[1:]
            if mask[-1]:
                on_inds = on_inds[:-1]
        else:
            if mask[0]:
                on_inds.insert(0, 0)
            if mask[-1]:
                off_inds.append(len(mask))

        if len(on_inds) == 0:
            continue
        
        for i in range(len(on_inds)):
            if on_inds[i] == off_inds[i]:
                continue
            hits.append((on_inds[i], off_inds[i]))
    
    ## sort hits  ## NOTE: this can be sped up since we already know how to interleave the events..
    hits.sort(key=lambda a: a[0])
    
    n_events = len(hits)
    events = np.empty(n_events, dtype=[
        ('index', int),
        ('len', int),
        ('sum', float),
        ('peak', float),
        ('peak_index', int),
        # only used if timing is available:
        ('time', float),
        ('duration', float),
        ('area', float),
        ('peak_time', float),
    ])
    
    ## Lots of work ahead:
    ## 1) compute length, peak, sum for each event
    ## 2) adjust event times if requested, then recompute parameters
    last_adj = 0
    for i in range(n_events):
        ind1, ind2 = hits[i]
        ln = ind2-ind1
        ev_data = data1[ind1:ind2]
        sum = ev_data.sum()
        if sum > 0:
            peak_ind = np.argmax(ev_data)
        else:
            peak_ind = np.argmin(ev_data)
        peak = ev_data[peak_ind]
        peak_ind += ind1
        
        #print "event %f: %d" % (xvals[ind1], ind1) 
        if adjust_times:  ## Move start and end times outward, estimating the zero-crossing point for the event
        
            ## adjust ind1 first
            mind = np.argmax(ev_data)
            pdiff = abs(peak - ev_data[0])
            if pdiff == 0:
                adj1 = 0
            else:
                adj1 = int(threshold * mind / pdiff)
                adj1 = min(ln, adj1)
            ind1 -= adj1
            
            ## check for collisions with previous events
            if i > 0:
                lind2 = hits[i-1][1]
                if ind1 < lind2:
                    diff = lind2-ind1   ## if events have collided, force them to compromise
                    tot = adj1 + last_adj
                    if tot != 0:
                        d1 = diff * float(last_adj) / tot
                        d2 = diff * float(adj1) / tot
                        hits[i-1] = (int(hits[i-1][0]), int(hits[i-1][1]-(d1+1)))
                        ind1 += d2
            
            ## adjust ind2
            mind = ln - mind
            pdiff = abs(peak - ev_data[-1])
            if pdiff == 0:
                adj2 = 0
            else:
                adj2 = int(threshold * mind / pdiff)
                adj2 = min(ln, adj2)
            ind2 += adj2
            last_adj = adj2
            
        hits[i] = (int(ind1), int(ind2))
        events[i]['peak'] = peak
        events[i]['index'] = ind1
        events[i]['peak_index'] = peak_ind
        events[i]['len'] = ln
        events[i]['sum'] = sum
        
    if adjust_times:  ## go back and re-compute event parameters.
        mask = np.ones(n_events, dtype=bool)
        for i in range(n_events):
            ind1, ind2 = hits[i]
            
            ln = ind2 - ind1
            ev_data = data1[ind1:ind2]
            sum = ev_data.sum()
            if len(ev_data) == 0:
                mask[i] = False
                continue
            if sum > 0:
                peak_ind = np.argmax(ev_data)
            else:
                peak_ind = np.argmin(ev_data)
            peak = ev_data[peak_ind]
            peak_ind += ind1
                
            events[i]['peak'] = peak
            events[i]['index'] = ind1
            events[i]['peak_index'] = peak_ind
            events[i]['len'] = ln
            events[i]['sum'] = sum
    
        ## remove masked events
        events = events[mask]

    # add in timing information if available:
    if trace.has_timing:
        for ev in events:
            i1 = ev['index']
            i2 = i1 + ev['len']
            ev['time'] = trace.time_at(i1)
            ev['duration'] = trace.time_at(i2) - ev['time']
            ev['area'] = np.trapz(y=data1[i1:i2], x=trace.time_values[i1:i2])
            ev['peak_time'] = trace.time_at(ev['peak_index'])
    else:
        ev['time'] = np.nan
        ev['duration'] = np.nan
        ev['area'] = np.nan
        ev['peak_time'] = np.nan

    return events


def rolling_sum(data, n):
    """A sliding-window filter that returns the sum of n source values for each
    value in the output array::
    
        output[i] = sum(input[i:i+n])
    
    """
    d1 = np.cumsum(data)
    d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
    d2[0] = d1[n-1]  # copy first point
    d2[1:] = d1[n:] - d1[:-n]  # subtract
    return d2
    

def clements_bekkers(data, template):
    """Sliding, scale-invariant template matching algorithm.

    Slides a template along a signal, measuring their similarity and scale difference
    at each sample. Similar to deconvolution.
    
    Parameters
    ----------
    data : array
        The signal data to process
    template : array
        A template 
    
    Returns
    -------
    detection_criterion : array
        Indicates how well the template matches the signal at each sample.
    scale : array
        The scale factor providing the best fit between template and signal at each sample.
    offset : array
        The y-offset of the best template fit at each sample.
    
    Notes
    -----
    Fast and sensitive for well-isolated events but performs poorly on overlapping events.
    See Clements & Bekkers, Biophysical Journal, 73: 220-229, 1997.
    """
    # Strip out meta-data for faster computation
    D = data.view(np.ndarray)
    T = template.view(np.ndarray)
    
    # Prepare a bunch of arrays we'll need later
    N = len(T)
    sumT = T.sum()
    sumT2 = (T**2).sum()
    sumD = rolling_sum(D, N)
    sumD2 = rolling_sum(D**2, N)
    sumTD = np.correlate(D, T, mode='valid')
    
    # compute scale factor, offset at each location:
    scale = (sumTD - sumT * sumD / N) / (sumT2 - sumT**2 / N)
    offset = (sumD - scale * sumT) / N
    
    # compute SSE at every location
    SSE = sumD2 + scale**2 * sumT2 + N * offset**2 - 2 * (scale*sumTD + offset*sumD - scale*offset*sumT)
    
    # finally, compute error and detection criterion
    error = np.sqrt(SSE / (N-1))
    DC = scale / error
    return DC, scale, offset


def exp_deconvolve(trace, tau):
    """Exponential deconvolution used to isolate overlapping events; works nicely on PSPs, calcium transients, etc.

    See: Richardson & Silberberg 2008, "Measurement and Analysis of Postsynaptic Potentials Using 
         a Novel Voltage-Deconvolution Method"
    """
    dt = trace.dt
    arr = trace.data
    deconv = arr[:-1] + (tau / dt) * np.diff(arr)
    if trace.has_time_values:
        # data is one sample shorter; clip time values to match.
        return trace.copy(data=deconv, time_values=trace.time_values[:-1])
    else:
        return trace.copy(data=deconv)
    
def exp_reconvolve(trace, tau):
    # equivalent to subtracting an exponential decay from the original unconvolved signal
    dt = trace.dt
    d = np.zeros(trace.data.shape, trace.data.dtype)
    dtt = dt / tau
    dtti = 1. - dtt
    for i in range(1, len(d)):
        d[i] = dtti * d[i-1] + dtt * trace.data[i-1]
    return trace.copy(data=d)


def exp_deconv_psp_params(amp, rise_time, rise_power, decay_tau):
    """Analytical solution to exponential deconvolution on a PSP shape.
    
    For parameter documentation, see fitting.psp.Psp.
    
    The *decay_tau* parameter is used both to describe the psp shape and as the deconvolution
    time constant. Under this constraint, the solution has exactly the same form as the original PSP
    but with different parameters.
    
    Returns
    -------
    deconv_amp : float
    deconv_rise_time : float
    deconv_rise_power : float
    deconv_decay_tau : float
    """
    # sympy helped us out a bit here:
    # import sympy
    # t, tr, td, rp, tdec = sympy.symbols('t tr td rp tdec')
    # psp = (1 - sympy.exp(-t/tr))**rp * sympy.exp(-t/td)
    # dec = sympy.simplify(psp + tdec * sympy.diff(psp, t))
    
    rise_tau = Psp._compute_rise_tau(rise_time, rise_power, decay_tau)
    deconv_decay_tau = 1 / (1 / rise_tau + 1 / decay_tau)
    deconv_rise_power = rise_power - 1
    deconv_rise_time = Psp._compute_rise_time(rise_tau, deconv_rise_power, deconv_decay_tau)
    old_max = Psp._psp_inner(rise_time, rise_tau, rise_power, decay_tau)
    new_max = Psp._psp_inner(deconv_rise_time, rise_tau, deconv_rise_power, deconv_decay_tau)
    deconv_amp = amp * new_max / old_max * rise_power * decay_tau / rise_tau
    return deconv_amp, deconv_rise_time, deconv_rise_power, deconv_decay_tau
