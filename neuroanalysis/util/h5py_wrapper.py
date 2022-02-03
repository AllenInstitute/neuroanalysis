import h5py
from h5py.h5t import check_string_dtype


def File(*args, **kwds):
    return H5pyWrapper(h5py.File(*args, **kwds))


class H5pyWrapper:
    """Wraps h5py objects to preserve string behavior from version 2.

    This allows the same code to use h5py version 2 or 3 without changes.
    """
    def __init__(self, obj):
        self.__dict__['_wrapped_obj'] = obj
        
    def __getattr__(self, name):
        
        return getattr(self._wrapped_obj, name)
    
    def __setattr__(self, name, value):
        setattr(self._wrapped_obj, name, value)
        
    def __getitem__(self, name):
        item = self._wrapped_obj[name]
        if isinstance(item, h5py.Dataset):
            if check_string_dtype(item.dtype):
                return item.asstr()
            else:
                return item
        else:
            return H5pyWrapper(item)

    def __setitem__(self, item, value):
        self._wrapped_obj[item] = value

    def __repr__(self):
        return f"<wrapper around {self._wrapped_obj} >"


if h5py.__version__.split('.')[0] == '2':
    def H5pyWrapper(obj):
        return obj
