import inspect
from pytest import raises
from neuroanalysis.util.optional_import import optional_import, OptionalImportError


def test_optional_import():
    # import numpy

    np = optional_import('numpy')
    assert inspect.ismodule(np)

    bad_mod = optional_import('aewhfjarelgsg')
    assert isinstance(bad_mod, OptionalImportError)
    with raises(ImportError):
        bad_mod.test

    # from numpy.random import normal
    
    norm = optional_import('numpy.random', 'normal')
    assert isinstance(norm(), float)

    bad_name = optional_import('numpy.random', 'aergjkseorser')
    assert isinstance(bad_name, OptionalImportError)
    with raises(ImportError):
        bad_name()

    # from numpy.random import normal, random

    norm, rand = optional_import('numpy.random', ['normal', 'random'])
    assert isinstance(norm(), float)
    assert isinstance(rand(), float)

    bad_names = optional_import('numpy.random', ['aergjkseorser', 'agfawreaerges'])
    for name in bad_names:
        assert isinstance(name, OptionalImportError)
        with raises(ImportError):
            name()

    # test again using package:

    norm = optional_import('.random', 'normal', package='numpy')
    assert isinstance(norm(), float)

    bad_name = optional_import('.random', 'aergjkseorser', package='numpy')
    assert isinstance(bad_name, OptionalImportError)
    with raises(ImportError):
        bad_name()

    norm, rand = optional_import('.random', ['normal', 'random'], package='numpy')
    assert isinstance(norm(), float)
    assert isinstance(rand(), float)

    bad_names = optional_import('.random', ['aergjkseorser', 'agfawreaerges'], package='numpy')
    for name in bad_names:
        assert isinstance(name, OptionalImportError)
        with raises(ImportError):
            name()






