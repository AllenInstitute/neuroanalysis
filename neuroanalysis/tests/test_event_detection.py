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
        

def test_exp_deconv_psp_params():
    from neuroanalysis.event_detection import exp_deconvolve, exp_deconv_psp_params
    from neuroanalysis.data import TSeries
    from neuroanalysis.fitting import Psp

    x = np.linspace(0, 0.02, 10000)
    amp = 1
    rise_time = 4e-3
    decay_tau = 10e-3
    rise_power = 2

    # Make a PSP waveform
    psp = Psp()
    y = psp.eval(x=x, xoffset=0, yoffset=0, amp=amp, rise_time=rise_time, decay_tau=decay_tau, rise_power=rise_power)

    # exponential deconvolution
    y_ts = TSeries(y, time_values=x)
    y_deconv = exp_deconvolve(y_ts, decay_tau).data

    # show that we can get approximately the same result using exp_deconv_psp_params
    d_amp, d_rise_time, d_rise_power, d_decay_tau = exp_deconv_psp_params(amp, rise_time, rise_power, decay_tau)
    y2 = psp.eval(x=x, xoffset=0, yoffset=0, amp=d_amp, rise_time=d_rise_time, decay_tau=d_decay_tau, rise_power=d_rise_power)

    assert np.allclose(y_deconv, y2[1:], atol=0.02)
