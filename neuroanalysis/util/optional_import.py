import importlib


def optional_import(module, fromlist=None, package=None):
    """Try importing a module, but if that fails, wait until the first time it is
    accessed before raising the ImportError.

    Parameters
    ----------
    module : str
        Name of module to import (or import from)
    fromlist : list of str
        Optional list of names to import 
    package : str
        Optional base package from which relative *module* names are constructed.
        This argument is required if the *module* is relative (begins with `.`).

    Examples::

        # import numba
        numba = optional_import('numba')

        # from numpy import array, zeros
        array, zeros = optional_import('numpy', names=['array', 'zeros'])

        # from ..mypackage import myname
        myname = optional_import('..mypackage', names=['myname'], package=__name__)

    """
    try:
        mod = importlib.import_module(module, package=package)
        if fromlist is not None:
            ret = []
            for name in fromlist:
                if hasattr(mod, name):
                    ret.append(getattr(mod, name))
                else:
                    ret.append(OptionalImportError(ImportError("cannot import name '%s' from '%s' (%s)" % (name, module, mod.__file__))))
            return ret
    except ImportError as exc:
        mod = OptionalImportError(exc)
        if fromlist is not None:
            return [mod] * len(fromlist)


class OptionalImportError(object):
    """Dummy object that just re-raises an ImportError when it is accessed.
    """
    def __init__(self, exc):
        self.exc = exc
    def __getattr__(self, attr):
        raise self.exc
    def __call__(self, *args, **kwds):
        raise self.exc

