try:
    from functools import lru_cache
except ImportError:
    # fake decorator; lru_cache only on python 3
    def lru_cache(*args, **kwds):
        return lambda fn: fn

