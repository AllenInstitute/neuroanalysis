import weakref


class WeakRef(object):
    """Replacement for weakref that allows reference to None.
    """
    def __init__(self, obj):
        if obj is None:
            self._ref = None
        else:
            self._ref = weakref.ref(obj)
            
    def __call__(self):
        if self._ref is None:
            return None
        else:
            obj = self._ref()
            if obj is None:
                raise RuntimeError("Referenced object has already been collected.")
            return obj
        
    @property
    def is_dead(self):
        if self._ref is None:
            return False
        else:
            return self._ref() is None
