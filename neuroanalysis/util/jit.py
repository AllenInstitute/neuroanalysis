import warnings
try:
    import numba
    have_numba = True
except ImportError:
    have_numba = False


no_numba_warn = True
use_numba = True


def _fake_jit(fn):
    return fn


def numba_jit(*args, **kwds):
    """Wrapper around numba.jit that fails gracefully if numba is not available.
    """
    global use_numba, have_numba, no_numba_warn
    if use_numba and have_numba:
        return numba.jit(*args, **kwds)
    else:
        if use_numba and no_numba_warn:
            warnings.warn("Could not import numba; falling back to slower implementation.")
            no_numba_warn = False
        return _fake_jit

