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
    d = TSeries(np.zeros(10), dt=0.1)
    
    assert len(threshold_events(d, 1)) == 0
    assert len(threshold_events(d, 0)) == 0
    
    d.data[5:7] = 6
    
    ev = threshold_events(d, 1)
    assert len(ev) == 1
    assert ev[0]['index'] == 5
    assert ev[0]['len'] == 2
    
    d.data[2:4] = -6
    ev = threshold_events(d, 1)
    expected = np.array([
        (2, 2, -12., -6., 2, 0.2, 0.2, -0.6, 0.2),
        (5, 2,  12.,  6., 5, 0.5, 0.2,  0.6, 0.5)],
        dtype=dtype
    )
    for k in ev.dtype.names:
        assert np.allclose(ev[k], expected[k])
        
    