import numpy as np
import warnings
from scipy.optimize import curve_fit

from .data import TSeries, PatchClampRecording
from .event_detection import threshold_events
from .filter import bessel_filter
from .util.data_test import DataTestCase


def detect_evoked_spikes(data, pulse_edges, **kwds):
    """Return a list of dicts describing spikes in a patch clamp recording that were evoked by a single stimulus pulse.

    This function simply wraps either detect_ic_evoked_spikes or detect_vc_evoked_spikes, depending on the clamp mode
    used in *data*.

    Parameters
    ==========
    data : PatchClampRecording
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *data*. 

    Returns
    =======
    spikes : list
        Each item in this list is a dictionary containing keys 'onset_time', 'max_slope_time', and 'peak_time',
        indicating three possible time points during the spike that could be detected. Any of these values may
        be None to indicate that the timepoint could not be reliably determined. Additional keys may be present,
        such as 'peak' and 'max_slope'.
    """
    trace = data['primary']
    if data.clamp_mode == 'vc':
        return detect_vc_evoked_spikes(trace, pulse_edges, **kwds)
    elif data.clamp_mode == 'ic':
        return detect_ic_evoked_spikes(trace, pulse_edges, **kwds)
    else:
        raise ValueError(f"Unsupported clamp mode {trace.clamp_mode}")


def rc_decay(t, tau, Vo): 
    """function describing the deriviative of the voltage.  If there
    is no spike one would expect this to fall off as the RC of the cell. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        return -(Vo/tau)*np.exp(-t/tau)


def detect_ic_evoked_spikes(trace, pulse_edges, dv2_threshold=40e3, mse_threshold=30., ui=None):
    """
    """
    if ui is not None:
        ui.clear()
        ui.console.setStack()
        ui.plt1.plot(trace.time_values, trace.data)

    assert trace.data.ndim == 1
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)
    
    # calculate derivatives within pulse window
    diff1 = trace.time_slice(*pulse_edges).diff()
    diff2 = diff1.diff()

    # mask out pulse artifacts in diff2 before lowpass filtering
    for edge in pulse_edges:
        apply_cos_mask(diff2, center=edge + 100e-6, radius=400e-6, power=2)

    # low pass filter the second derivative
    diff2 = bessel_filter(diff2, 10e3, order=4, bidir=True)

    # look for positive bumps in second derivative
    events2 = list(threshold_events(diff2 / dv2_threshold, threshold=1.0, adjust_times=False))

    if ui is not None:
        ui.plt2.plot(diff1.time_values, diff1.data)
        ui.plt3.plot(diff2.time_values, diff2.data)
        ui.plt3.addLine(y=dv2_threshold)
    
    # for each bump in d2, either discard the event or generate spike metrics
    spikes = []
    for ev in events2:
        total_area = ev['area']
        onset_time = ev['time']

        # ignore events near pulse offset
        if abs(onset_time - pulse_edges[1]) < 50e-6:
            continue

        # require dv2 bump to be positive, not tiny
        if total_area < 10e-6:
            continue
        
        # don't double-count spikes within 1 ms
        if len(spikes) > 0 and onset_time < spikes[-1]['onset_time'] + 1e-3:
            continue

        max_slope_window = onset_time, pulse_edges[1]-50e-6
        max_slope_chunk = diff1.time_slice(*max_slope_window)
        if len(max_slope_chunk) == 0:
            continue
        max_slope_idx = np.argmax(max_slope_chunk.data)
        max_slope_time = max_slope_chunk.time_at(max_slope_idx)

        max_slope_time, is_edge = max_time(diff1.time_slice(onset_time, pulse_edges[1] - 50e-6))
        max_slope = diff1.value_at(max_slope_time)
        # require dv/dt to be above a threshold value
        if max_slope <= 30:  # mV/ms
            continue
        if is_edge != 0:
            # can't see max slope
            max_slope_time = None
            max_slope = None
        peak_time, is_edge = max_time(trace.time_slice(onset_time, pulse_edges[1] + 2e-3))
        if is_edge != 0 or pulse_edges[1] < peak_time < pulse_edges[1] + 50e-6:
            # peak is obscured by pulse edge
            peak_time = None
        
        spikes.append({
            'onset_time': onset_time,
            'peak_time': peak_time,
            'max_slope_time': max_slope_time,
            'peak_value': None if peak_time is None else trace.value_at(peak_time),
            'max_slope': max_slope,
        })

    # if no spike was found in the pulse region check to see if there is a spike in the pulse termination region
    if len(spikes) == 0:
        # note that this is using the dvdt with the termination artifact in it to locate where it should start 
        dv_after_pulse = trace.time_slice(pulse_edges[1] + 100e-6, None).diff()
        dv_after_pulse = bessel_filter(dv_after_pulse, 15e3, bidir=True)

        if dv_after_pulse.duration > 0.5e-3:
            # create a vector to fit
            ttofit = dv_after_pulse.time_values  # setting time to start at zero, note: +1 because time trace of derivative needs to be one shorter
            ttofit = ttofit - ttofit[0]

            # do fit and see if it matches
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    
                popt, pcov = curve_fit(rc_decay, ttofit, dv_after_pulse.data, maxfev=10000)
            fit = rc_decay(ttofit, *popt)
            if ui is not None:
                ui.plt2.plot(dv_after_pulse.time_values, dv_after_pulse.data)
                ui.plt2.plot(dv_after_pulse.time_values, fit, pen='b')

            diff = dv_after_pulse - fit
            mse = (diff.data**2).mean()  # mean squared error
            if mse > mse_threshold:
                search_window = 2e-3
                max_slope_time, is_edge = max_time(diff.time_slice(pulse_edges[1], pulse_edges[1] + search_window))
                if is_edge != 0:
                    max_slope_time = None
                peak_time, is_edge = max_time(trace.time_slice(max_slope_time or pulse_edges[1] + 100e-6, pulse_edges[1] + search_window))
                if is_edge != 0:
                    peak_time = None
                spikes.append({
                    'onset_time': None,
                    'max_slope_time': max_slope_time,
                    'peak_time': peak_time,
                    'peak_value': None if peak_time is None else trace.value_at(peak_time),
                    'max_slope': None if max_slope_time is None else dv_after_pulse.value_at(max_slope_time),
                })

    for spike in spikes:
        assert 'max_slope_time' in spike
    return spikes


def detect_vc_evoked_spikes(trace, pulse_edges, ui=None):
    """Return a dict describing an evoked spike in a patch clamp recording, or None if no spike is detected.

    This function assumes that a square voltage pulse is used to evoke an unclamped spike
    in a voltage clamp recording, and that the peak of the unclamped spike occurs *during*
    the stimulation pulse.

    Parameters
    ==========
    trace : TSeries instance
        The recorded patch clamp data. The recording should be made with a brief pulse intended
        to evoke a single spike with short latency.
    pulse_edges : (float, float)
        The start and end times of the stimulation pulse, relative to the timebase in *trace*.
    """
    if not isinstance(trace, TSeries):
        raise TypeError("data must be TSeries instance.")

    if ui is not None:
        ui.clear()
        ui.console.setStack()
        ui.plt1.plot(trace.time_values, trace.data)

    assert trace.ndim == 1
    pulse_edges = tuple(map(float, pulse_edges))  # make sure pulse_edges is (float, float)

    diff1 = trace.time_slice(pulse_edges[0], pulse_edges[1] + 2e-3).diff()
    diff2 = diff1.diff()

    # crop and filter diff1
    diff1 = diff1.time_slice(pulse_edges[0] + 100e-6, pulse_edges[1])
    diff1 = bessel_filter(diff1, cutoff=20e3, order=4, btype='low', bidir=True)

    # crop and low pass filter the second derivative
    diff2 = diff2.time_slice(pulse_edges[0] + 150e-6, pulse_edges[1])
    diff2 = bessel_filter(diff2, 20e3, order=4, bidir=True)
    # chop off ending transient
    diff2 = diff2.time_slice(None, diff2.t_end - 100e-6)

    # look for negative bumps in second derivative
    # dv1_threshold = 1e-6
    dv2_threshold = 0.02
    events = list(threshold_events(diff2 / dv2_threshold, threshold=1.0, adjust_times=False, omit_ends=True))

    if ui is not None:
        ui.plt2.plot(diff1.time_values, diff1.data)
        # ui.plt2.plot(diff1_hp.time_values, diff1.data)
        # ui.plt2.addLine(y=-dv1_threshold)
        ui.plt3.plot(diff2.time_values, diff2.data)
        ui.plt3.addLine(y=dv2_threshold)

    if len(events) == 0:
        return []

    spikes = []
    for ev in events:
        if ev['sum'] > 0 and ev['peak'] < 5. and ev['time'] < diff2.t0 + 60e-6:
            # ignore positive bumps very close to the beginning of the trace
            continue
        if len(spikes) > 0 and ev['peak_time'] < spikes[-1]['max_slope_time'] + 1e-3:
            # ignore events that follow too soon after a detected spike
            continue

        if ev['sum'] < 0:
            onset_time = ev['peak_time']
            search_time = onset_time
        else:
            search_time = ev['time'] - 200e-6
            onset_time = None

        max_slope_rgn = diff1.time_slice(search_time, search_time + 0.5e-3)
        max_slope_time, is_edge = min_time(max_slope_rgn)
        max_slope = diff1.value_at(max_slope_time)
        if max_slope > 0:
            # actual slope must be negative at this point
            # (above we only tested the sign of the high-passed event)
            continue
        
        peak_search_rgn = trace.time_slice(max_slope_time, min(pulse_edges[1], search_time + 1e-3))
        if len(peak_search_rgn) == 0:
            peak = None
            peak_time = None
        else:
            peak_time, is_edge = min_time(peak_search_rgn)
            if is_edge:
                peak = None
                peak_time = None
            else:
                peak = trace.time_at(peak_time)

        spikes.append({
            'onset_time': onset_time,
            'max_slope_time': max_slope_time,
            'max_slope': max_slope,
            'peak_time': peak_time,
            'peak_value': peak,
        })

    for spike in spikes:
        assert 'max_slope_time' in spike
    return spikes


def apply_cos_mask(trace, center, radius, power):
    """Multiply a region of a trace by a cosine mask to dampen artifacts without generating 
    sharp edges.
    
    The input *trace* is modified in-place.
    """
    chunk = trace.time_slice(center - radius, center + radius)
    w1 = np.pi * (chunk.t0 - center) / radius
    w2 = np.pi * (chunk.t_end - center) / radius
    mask_t = np.pi + np.linspace(w1, w2, len(chunk))
    mask = ((np.cos(mask_t) + 1) * 0.5) ** power
    chunk.data[:] = chunk.data * mask


def max_time(trace):
    """Return the time of the maximum value in the trace, and a value indicating whether the
    time returned coincides with the first value in the trace (-1), the last value in the
    trace (1) or neither (0).
    """
    ind = np.argmax(trace.data)
    if ind == 0:
        is_edge = -1
    elif ind == len(trace) - 1:
        is_edge = 1
    else:
        is_edge = 0
    return trace.time_at(ind), is_edge


def min_time(trace):
    """Return the time of the minimum value in the trace, and a value indicating whether the
    time returned coincides with the first value in the trace (-1), the last value in the
    trace (1) or neither (0).
    """
    ind = np.argmin(trace.data)
    if ind == 0:
        is_edge = -1
    elif ind == len(trace) - 1:
        is_edge = 1
    else:
        is_edge = 0
    return trace.time_at(ind), is_edge


class SpikeDetectTestCase(DataTestCase):
    def __init__(self):
        DataTestCase.__init__(self, detect_evoked_spikes)

    def check_result(self, result):
        for spike in result:
            assert 'max_slope_time' in spike
            assert 'onset_time' in spike
            assert 'peak_time' in spike
        DataTestCase.check_result(self, result)

    @property
    def name(self):
        meta = self.meta
        return "%s_%s_%s_%0.3f" % (meta['expt_id'], meta['sweep_id'], meta['device_id'], self.input_args['pulse_edges'][0])
