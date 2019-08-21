import importlib


def optional_import(module):
    """Try importing a module, but if that fails, wait until the first time it is
    accessed before raising the ImportError.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        return OptionalImportError(exc)


class OptionalImportError(object):
    def __init__(self, exc):
        self.exc = exc
    def __getattr__(self, attr):
        raise self.exc
