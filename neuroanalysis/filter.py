import numpy as np
import scipy.stats, scipy.signal


def bessel_filter(trace, cutoff, order=1, btype='low', bidir=True):
    """Return a Bessel-filtered copy of a TSeries.
    """
    b,a = scipy.signal.bessel(order, cutoff * trace.dt, btype=btype) 
    filtered = apply_filter(trace.data, b, a, bidir=bidir)
    # todo: record information about filtering?
    #filtered.meta['processing'].append({'name': 'bessel_filter', 'cutoff': cutoff, 'order': order, 'btype': btype, 'bidir': bidir})
    return trace.copy(data=filtered)


def butterworth_filter(trace, w_pass, w_stop=None, g_pass=2.0, g_stop=20.0, order=1, btype='low', bidir=True):
    """Return a Butterworth-filtered copy of a TSeries.
    """
    if w_stop is None:
        w_stop = w_pass * 2.0
    dt = trace.dt
    ord, Wn = scipy.signal.buttord(w_pass*dt*2., w_stop*dt*2., g_pass, g_stop)
    b,a = scipy.signal.butter(ord, Wn, btype=btype) 
    filtered = apply_filter(trace.data, b, a, bidir=bidir)

    return trace.copy(data=filtered)


def apply_filter(data, b, a, padding=100, bidir=True):
    """Apply a linear filter with coefficients a, b. Optionally pad the data before filtering
    and/or run the filter in both directions.
    """
    if padding > 0:
        pad1 = data[:padding][::-1]
        pad2 = data[-padding:][::-1]
        data = np.hstack([pad1, data, pad2])
    
    if bidir:
        filtered = scipy.signal.lfilter(b, a, scipy.signal.lfilter(b, a, data)[::-1])[::-1]
    else:
        filtered = scipy.signal.lfilter(b, a, data)
    
    if padding > 0:
        filtered = filtered[len(pad1):-len(pad2)]
        
    return filtered

def savgol_filter(trace, window_duration, **kwds):
    """Return a Savitsky-Golay-filtered copy of a TSeries.
    """
    from scipy.signal import savgol_filter  # only recently available
    wlen = int(window_duration / trace.dt)
    filtered = savgol_filter(trace.data, window_length=wlen, **kwds)
    return trace.copy(data=filtered)
    

def remove_artifacts(trace, edges, window):
    """Remove selected regions from a trace and fill with a flat line.

    Parameters
    ----------
    trace : TSeries instance
        Data to be filtered.
    edges : list of (start, stop) tuples
        Specifies the indices of regions in the trace to remove.
    window : float
        Window duration (in seconds) on either side of each removed
        chunk that will be used to determine the values used to fill
        the chunk.

    Returns
    -------
    A copy of *trace* with artifacts removed.

    Notes
    -----
    For each (start, stop) pair in *edges*, three windows are used:

    1. The window in [start:stop] that will have its values replaced
    2. The window immediately before (1), of width determined by the *window* argument
    3. The window immediately after (1), of width determined by the *window* argument

    The values in (1) are replaced by performing a linear regression on the data in
    (2) and (3), then filling (1) with the resulting extrapolated line.
    """
    data = trace.data.copy()
    t = trace.time_values
    w = int(window / trace.dt)
    
    # merge together overlapping regions
    edges.sort()
    merged_edges = [edges[0]]
    for on, off in edges[1:]:
        on1, off1 = merged_edges[-1]
        if on < off1:
            merged_edges[-1] = (min(on, on1), max(off, off1))
        else:
            merged_edges.append((on, off))

    # remove and replace with linregress
    for on, off in merged_edges:
        on = int(on)
        off = int(off)
        chunkx = t[on-w:off+w]
        chunky = data[on-w:off+w]
        mask = np.ones(len(chunkx), dtype='bool')
        mask[w:w+(off-on)] = False
        chunkx = chunkx[mask]
        chunky = chunky[mask]
        slope, intercept = scipy.stats.linregress(chunkx, chunky)[:2]
        data[on:off] = slope * t[on:off] + intercept
    
    return trace.copy(data=data)


def downsample(data, n, axis=0):
    """Downsample *data* by averaging *n* points together across *axis*.
    """
    n = int(n)
    if n == 1:
        return data
    elif n < 1:
        raise ValueError("Invalid downsampling window %d" % n)
    
    n_pts = int(data.shape[axis] / n)
    s = list(data.shape)
    s[axis] = n_pts
    s.insert(axis+1, n)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(0, n_pts*n)
    d1 = data[tuple(sl)]
    d1.shape = tuple(s)
    d2 = d1.mean(axis+1)

    return d2
