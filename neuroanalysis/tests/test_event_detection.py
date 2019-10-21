from pytest import raises
import numpy as np

from neuroanalysis.data import TSeries
from neuroanalysis.event_detection import threshold_events

dtype = [
    ('index', int),
    ('len', int),
    ('sum', float),
    ('peak', float),
    ('peak_index', int),
    ('time', float),
    ('duration', float),
    ('area', float),
    ('peak_time', float),
]

def test_threshold_events():
    empty_result = np.array([], dtype=dtype)

    d = TSeries(np.zeros(10), dt=0.1)
    
    check_events(threshold_events(d, 1), empty_result)
    check_events(threshold_events(d, 0), empty_result)
    
    d.data[5:7] = 6
    
    ev = threshold_events(d, 1)
    expected = np.array([(5, 2, 12., 6., 5, 0.5, 0.2, 0.6, 0.5)], dtype=dtype)
    check_events(threshold_events(d, 1), expected)
    
    d.data[2:4] = -6
    expected = np.array([
        (2, 2, -12., -6., 2, 0.2, 0.2, -0.6, 0.2),
        (5, 2,  12.,  6., 5, 0.5, 0.2,  0.6, 0.5)],
        dtype=dtype
    )
    check_events(threshold_events(d, 1), expected)
        
    # data ends above threshold
    d.data[:] = 0
    d.data[5:] = 6
    check_events(threshold_events(d, 1), empty_result)
    expected = np.array([(5, 5, 30., 6., 5, 0.5, 0.5, 2.4, 0.5)], dtype=dtype)
    check_events(threshold_events(d, 1, omit_ends=False), expected)

    # data begins above threshold
    d.data[:] = 6
    d.data[5:] = 0
    check_events(threshold_events(d, 1), empty_result)
    expected = np.array([(0, 5, 30., 6., 0, 0., 0.5, 2.4, 0.)], dtype=dtype)    
    check_events(threshold_events(d, 1, omit_ends=False), expected)

    # all points above threshold
    d.data[:] = 6
    check_events(threshold_events(d, 1), empty_result)
    expected = np.array([(0, 10, 60., 6., 0, 0., 1., 5.4, 0.)], dtype=dtype)
    check_events(threshold_events(d, 1, omit_ends=False), expected)
    

def check_events(a, b):
    # print("Check:")
    # print("np.array(%s, dtype=dtype)" % a)
    # print("Expected:")
    # print("np.array(%s, dtype=dtype)" % b)
    assert(a.dtype == b.dtype)
    assert(a.shape == b.shape)
    for k in a.dtype.names:
        assert np.allclose(a[k], b[k])
        