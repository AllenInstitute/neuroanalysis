import json
from collections import OrderedDict

import numpy as np
import pytest

import neuroanalysis.stimuli as stimuli
from neuroanalysis.data.dataset import TSeries


def test_stimulus():
    # test a generic stimulus
    s1 = stimuli.Stimulus("stimulus 1")
    assert s1.description == "stimulus 1"
    assert s1.global_start_time == 0
    assert s1.start_time == 0
    assert s1.type == "Stimulus"
    assert s1.parent is None
    assert len(s1.items) == 0

    # test save/load
    state = s1.save()
    assert state == OrderedDict([
        ('type', 'Stimulus'),
        ('args', OrderedDict([
            ('start_time', 0), 
            ('description', 'stimulus 1'),
            ('units', None),
        ])),
        ('items', []),
    ])
    s2 = stimuli.load_stimulus(state)
    assert isinstance(s2, stimuli.Stimulus)
    assert s2.description == "stimulus 1"
    assert s2.global_start_time == 0
    assert s2.start_time == 0
    assert s2.type == "Stimulus"
    assert s2.parent is None
    assert len(s2.items) == 0


def test_square_pulse():
    # test SquarePulse
    sp1 = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=10)
    assert sp1.type == "SquarePulse"
    assert sp1.description == 'square pulse'
    assert sp1.parent is None
    assert len(sp1.items) == 0
    assert sp1.global_start_time == 0.1
    assert sp1.start_time == 0.1
    assert sp1.duration == 0.2
    assert sp1.amplitude == 10

    # check evaluation of pulse waveform
    sp1_eval = sp1.eval(n_pts=1000, dt=0.001)
    assert sp1_eval.dt == 0.001
    sp1_data = sp1_eval.data
    assert sp1_data.shape == (1000,)
    test_data = np.zeros(1000)
    test_data[100:300] = 10
    assert np.all(sp1_data == test_data)
    assert np.all((sp1_data != 0) == sp1.mask(time_values=sp1_eval.time_values).data)

    # test parent/child logic
    s1 = stimuli.Stimulus("stimulus 1")
    sp1.parent = s1
    assert sp1.parent is s1
    assert s1.items == (sp1,)

    sp1.parent = None
    assert sp1.parent is None
    assert s1.items == ()
    
    s1.append_item(sp1)
    assert sp1.parent is s1
    assert s1.items == (sp1,)
    
    s1.remove_item(sp1)
    assert sp1.parent is None
    assert s1.items == ()

    # add in a second pulse
    sp2 = stimuli.SquarePulse(start_time=0.2, duration=0.2, amplitude=-10, description="square pulse 2", parent=s1, units='A')
    assert sp2.description == "square pulse 2"
    assert sp2.parent is s1
    assert s1.items == (sp2,)
    
    # more parent / child logic testing
    s1.insert_item(1, sp1)
    assert sp1.parent is s1
    assert s1.items == (sp2, sp1)

    s1.remove_item(sp1)
    s1.insert_item(0, sp1)
    assert sp1.parent is s1
    assert s1.items == (sp1, sp2)

    # test waveform eval
    sp2_eval = sp2.eval(n_pts=1000, dt=0.001)
    assert sp2_eval.units == 'A'
    sp2_test_data = np.zeros(1000)
    sp2_test_data[200:400] = -10
    assert np.all(sp2_test_data == sp2_eval.data)

    # test waveform eval with two pulses
    s1_data = s1.eval(n_pts=1000, dt=0.001).data
    test_data[200:400] -= 10
    assert np.all(s1_data == test_data)

    # test save/load
    state = sp1.save()
    assert state == OrderedDict([
        ('type', 'SquarePulse'),
        ('args', OrderedDict([
            ('start_time', 0.1), 
            ('description', 'square pulse'),
            ('units', None),
            ('duration', 0.2),
            ('amplitude', 10),
        ])),
        ('items', []),
    ])
    sp3 = stimuli.load_stimulus(state)
    assert isinstance(sp3, stimuli.SquarePulse)
    assert sp3.type == sp1.type == "SquarePulse"
    assert sp3.description == sp1.description == 'square pulse'
    assert sp3.global_start_time == sp1.global_start_time == 0.1
    assert sp3.start_time == sp1.start_time == 0.1
    assert sp3.duration == sp1.duration == 0.2
    assert sp3.amplitude == sp1.amplitude == 10
    assert sp3.parent is None
    assert len(sp3.items) == 0


def test_pulse_train():
    s1 = stimuli.Stimulus("stimulus 1")
    sp1 = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=10, parent=s1)
    sp2 = stimuli.SquarePulse(start_time=0.2, duration=0.2, amplitude=-10, description="square pulse 2", units='A', parent=s1)

    # test PulseTrain
    pt1 = stimuli.SquarePulseTrain(start_time=0.5, n_pulses=3, pulse_duration=0.02, interval=0.1, amplitude=1.0, parent=s1, units='A')
    assert s1.items == (sp1, sp2, pt1)

    # check sub-pulse start times
    assert pt1.global_start_time == 0.5
    assert pt1.start_time == 0.5
    assert pt1.items[0].global_start_time == 0.5
    assert pt1.items[0].start_time == 0.0
    assert pt1.items[1].global_start_time == 0.6
    assert pt1.items[1].start_time == 0.1
    assert pt1.items[2].global_start_time == 0.7
    assert pt1.items[2].start_time == 0.2
    assert pt1.pulse_times == [0.0, 0.1, 0.2]
    assert pt1.global_pulse_times == [0.5, 0.6, 0.7]

    # test waveform eval with all three items
    s1_data = s1.eval(n_pts=1000, dt=0.001).data
    test_data = np.zeros(1000)
    test_data[100:300] = 10
    test_data[200:400] -= 10
    test_data[500:520] += 1
    test_data[600:620] += 1
    test_data[700:720] += 1
    assert np.all(s1_data == test_data)

    test_mask = (s1_data != 0)
    test_mask[100:400] = True
    assert np.all(test_mask == s1.mask(n_pts=1000, dt=0.001).data)

    # test save/load
    state = s1.save()
    assert state == OrderedDict([
        ('type', 'Stimulus'),
        ('args', OrderedDict([
            ('start_time', 0),
            ('description', 'stimulus 1'),
            ('units', None),
        ])),
        ('items', [
            OrderedDict([
                ('type', 'SquarePulse'),
                ('args', OrderedDict([
                    ('start_time', 0.1), 
                    ('description', 'square pulse'),
                    ('units', None),
                    ('duration', 0.2),
                    ('amplitude', 10),
                ])),
                ('items', []),
            ]),
            OrderedDict([
                ('type', 'SquarePulse'),
                ('args', OrderedDict([
                    ('start_time', 0.2), 
                    ('description', 'square pulse 2'),
                    ('units', 'A'),
                    ('duration', 0.2),
                    ('amplitude', -10),
                ])),
                ('items', []),
            ]),
            OrderedDict([
                ('type', 'SquarePulseTrain'),
                ('args', OrderedDict([
                    ('start_time', 0.5), 
                    ('description', 'square pulse train'),
                    ('units', 'A'),
                    ('n_pulses', 3),
                    ('pulse_duration', 0.02),
                    ('amplitude', 1.0),
                    ('interval', 0.1),
                ])),
                ('items', []),
            ]),
        ])
    ])
    s2 = stimuli.load_stimulus(state)
    assert isinstance(s2, stimuli.Stimulus)
    assert s2.items[0] == s1.items[0]
    assert s2.items[1] == s1.items[1]
    assert s2.items[2] == s1.items[2]
    assert s2.items[2].items[0] == s1.items[2].items[0]
    assert s2.items[2].items[1] == s1.items[2].items[1]
    assert s2.items[2].items[2] == s1.items[2].items[2]
    

def test_offset():
    stim = stimuli.Offset(start_time=1.0, amplitude=2.3)
    t = stim.eval(t0=0, n_pts=1000, dt=0.01)
    test_data = np.zeros(1000)
    test_data[100:] = 2.3
    assert np.all(t.data == test_data)
    assert np.all(stim.mask(time_values=t.time_values).data == (t.data != 0))


def test_ramp():
    stim = stimuli.Ramp(start_time=1.0, slope=2.3, duration=1.5, offset=0.5)
    t = stim.eval(t0=0, n_pts=1000, dt=0.01)
    test_data = np.zeros(1000)
    test_data[100:250] = 0.5 + np.arange(150) * 0.01 * 2.3
    assert np.allclose(t.data, test_data)
    assert np.all(stim.mask(time_values=t.time_values).data == (t.data != 0))


def test_sine():
    stim = stimuli.Sine(start_time=1.0, frequency=12.7, duration=1.5, amplitude=0.1, offset=0.5, phase=2.1)
    t = stim.eval(t0=0, n_pts=1000, dt=0.01)
    test_data = np.zeros(1000)
    w = 2 * np.pi * 12.7 * np.arange(100,250)*0.01
    test_data[100:250] = 0.5 + 0.1 * np.sin(w + 2.1 - w[0])
    assert np.allclose(t.data, test_data)
    assert np.all(stim.mask(time_values=t.time_values).data == (t.data != 0))


def test_chirp():
    f0 = 13.5
    f1 = f0 * 100
    dur = 0.8
    amp = 0.1
    off = 0.5
    ph = 2.1
    stim = stimuli.Chirp(start_time=0.1, start_frequency=f0, end_frequency=f1, duration=dur, amplitude=amp, offset=off, phase=ph)
    npts = 10000
    dt = 0.0001
    eval_data = stim.eval(t0=0, n_pts=npts, dt=dt).data
    mask_data = stim.mask(t0=0, n_pts=npts, dt=dt).data
    test_data = np.zeros(npts)

    # sin[2 pi f(0) d (f(d) / f(0))^(t / d) / ln(f(d) / f(0)]
    t = np.arange(8000) * dt
    w = 2 * np.pi * f0 * dur * (f1/f0)**(t/dur) / np.log(f1/f0)
    test_data[1000:9000] = off + amp * np.sin(w + ph - w[0])
    assert np.allclose(eval_data, test_data)
    assert np.all(mask_data == (eval_data != 0))

    # test eval with first timepoint not at beginning of chirp
    t2 = np.arange(5000, 10000) * dt
    assert np.allclose(stim.eval(time_values=t2).data, eval_data[5000:])

    # measure approximate frequency by interpolated zero-crossings
    d = eval_data[1000:9000]-off
    inds = np.where(np.diff(np.sign(d)))[0]
    d0 = d[inds]
    d1 = d[inds+1]
    s = abs(d0) / abs(d1-d0)
    zeros = (inds + s) * dt
    freqs = 0.5 / np.diff(zeros)

    # check endpoints
    assert (abs(freqs[0] / f0) - 1.0) < 0.3
    assert (abs(freqs[-1] / f1) - 1.0) < 0.1

    # check center
    ind = np.argwhere(zeros > 0.4)[0,0]
    assert (abs(freqs[ind] / (f0*10)) - 1.0) < 0.1

    # test analytically determined frequencies
    freqs = f0 ** np.linspace(1, np.log(f1) / np.log(f0), len(t)+1)[:-1]
    assert np.allclose(freqs, stim.frequency_at(t))

def test_validation():
    # Sine: Nyquist check in eval
    sine = stimuli.Sine(start_time=0, duration=0.1, frequency=100, amplitude=1)
    with pytest.raises(ValueError):
        sine.eval(n_pts=100, sample_rate=150)  # 150 < 2*100
    sine.eval(n_pts=100, sample_rate=300)  # should be fine

    # Sine: frequency must be positive
    with pytest.raises(ValueError):
        stimuli.Sine(start_time=0, duration=0.1, frequency=0, amplitude=1)
    with pytest.raises(ValueError):
        stimuli.Sine(start_time=0, duration=0.1, frequency=-5, amplitude=1)

    # Sine: duration must be non-negative
    with pytest.raises(ValueError):
        stimuli.Sine(start_time=0, duration=-1, frequency=10, amplitude=1)

    # Chirp: Nyquist check in eval (restore original check)
    chirp = stimuli.Chirp(start_time=0, duration=0.1, start_frequency=10, end_frequency=100, amplitude=1)
    with pytest.raises(ValueError):
        chirp.eval(n_pts=100, sample_rate=150)  # 150 < 2*100
    chirp.eval(n_pts=100, sample_rate=300)  # should be fine

    # Chirp: frequencies must be positive
    with pytest.raises(ValueError):
        stimuli.Chirp(start_time=0, duration=0.1, start_frequency=0, end_frequency=100, amplitude=1)
    with pytest.raises(ValueError):
        stimuli.Chirp(start_time=0, duration=0.1, start_frequency=10, end_frequency=-1, amplitude=1)

    # Chirp: duration must be positive
    with pytest.raises(ValueError):
        stimuli.Chirp(start_time=0, duration=0, start_frequency=10, end_frequency=100, amplitude=1)
    with pytest.raises(ValueError):
        stimuli.Chirp(start_time=0, duration=-1, start_frequency=10, end_frequency=100, amplitude=1)

    # SquarePulse: duration must be non-negative
    with pytest.raises(ValueError):
        stimuli.SquarePulse(start_time=0, duration=-0.1, amplitude=1)

    # SquarePulseTrain: n_pulses must be a positive integer
    with pytest.raises(ValueError):
        stimuli.SquarePulseTrain(start_time=0, n_pulses=0, pulse_duration=0.01, amplitude=1, interval=0.1)
    with pytest.raises(ValueError):
        stimuli.SquarePulseTrain(start_time=0, n_pulses=-3, pulse_duration=0.01, amplitude=1, interval=0.1)
    with pytest.raises(TypeError):
        stimuli.SquarePulseTrain(start_time=0, n_pulses=1.5, pulse_duration=0.01, amplitude=1, interval=0.1)

    # SquarePulseTrain: interval must be >= pulse_duration
    with pytest.raises(ValueError):
        stimuli.SquarePulseTrain(start_time=0, n_pulses=3, pulse_duration=0.1, amplitude=1, interval=0.05)

    # Ramp: duration must be non-negative
    with pytest.raises(ValueError):
        stimuli.Ramp(start_time=0, duration=-1, slope=1)

    # Psp: rise_time must be positive
    with pytest.raises(ValueError):
        stimuli.Psp(start_time=0, rise_time=0, decay_tau=0.01, amplitude=1)
    with pytest.raises(ValueError):
        stimuli.Psp(start_time=0, rise_time=-0.01, decay_tau=0.01, amplitude=1)

    # Psp: decay_tau must be positive
    with pytest.raises(ValueError):
        stimuli.Psp(start_time=0, rise_time=0.01, decay_tau=0, amplitude=1)
    with pytest.raises(ValueError):
        stimuli.Psp(start_time=0, rise_time=0.01, decay_tau=-0.01, amplitude=1)

    # Psp: rise_power must be positive
    with pytest.raises(ValueError):
        stimuli.Psp(start_time=0, rise_time=0.01, decay_tau=0.01, amplitude=1, rise_power=0)
    with pytest.raises(ValueError):
        stimuli.Psp(start_time=0, rise_time=0.01, decay_tau=0.01, amplitude=1, rise_power=-1)

    # SquarePulseSeries: mismatched array lengths should raise ValueError (not AssertionError)
    with pytest.raises(ValueError):
        stimuli.SquarePulseSeries(start_time=0, pulse_times=[0, 0.1], pulse_durations=[0.01], amplitudes=[1, 1])


def test_find_noisy_square_pulses():
    dt = 0.0002
    np.random.seed(54321)

    data = np.random.normal(0.002, 0.0015, 10000)
    ## add a small 60 Hz sine wave
    data += 0.004 * np.sin(np.arange(10000) * 2.0 * np.pi / (1/60. * 1/dt))

    ## create a pulse an make the edges a little fuzzy
    amp1 = 1
    data[2001:2200] += amp1
    data[2000] += (0.7 * amp1)
    data[2200] += (0.2 * amp1)

    ## create a second smaller pulse
    amp2 = 0.3
    data[5000:5500] += amp2

    tseries = TSeries(data=data, dt=dt, units='V')

    pulses = stimuli.find_noisy_square_pulses(tseries)

    ## make sure we found correct number and that they are SquarePulses
    assert len(pulses) == 2
    assert isinstance(pulses[0], stimuli.SquarePulse)


    ## check parameters
    assert pulses[0].start_time == 0.4
    assert np.isclose(pulses[0].amplitude, amp1, 0.001, 0.001)
    assert pulses[0].duration == 0.04
    assert pulses[1].start_time == 1
    assert np.isclose(pulses[1].amplitude, amp2, 0.001, 0.001)
    assert pulses[1].duration == 0.1


# --- helpers for the tests below ---

DT = 0.001
N_PTS = 2000


def _eval(stim, dt=DT, n_pts=N_PTS):
    return stim.eval(t0=0, n_pts=n_pts, dt=dt).data


def _check_round_trip(stim):
    """Check that save/load (direct and via JSON) preserves equality and eval output; return the loaded copy."""
    state = stim.save()
    json_state = json.loads(json.dumps(state))
    assert json_state == state
    expected = _eval(stim)
    for st in (state, json_state):
        loaded = stimuli.Stimulus.load(st)
        assert type(loaded) is type(stim)
        assert loaded == stim
        assert loaded.parent is None
        assert loaded.save() == state
        assert np.array_equal(_eval(loaded), expected)
    return loaded


class FakeLoader:
    """Minimal loader for LazyLoadStimulus; records each source it is asked to load."""
    def __init__(self, items_by_source):
        self.items_by_source = items_by_source
        self.calls = []

    def load_stimulus_items(self, source):
        self.calls.append(source)
        return self.items_by_source[source]


class CustomPulse(stimuli.SquarePulse):
    pass


class CustomPulseVariant(CustomPulse):
    pass


# --- per-class parameter, eval, and save/load tests ---

def test_stimulus_params():
    child = stimuli.SquarePulse(start_time=0.1, duration=0.05, amplitude=2.0)
    s = stimuli.Stimulus(description="base", start_time=0.25, units='V', items=[child])
    assert s.description == "base"
    assert s.start_time == 0.25
    assert s.global_start_time == 0.25
    assert s.units == 'V'
    assert s.duration == 0
    assert s.items == (child,)
    assert child.parent is s
    assert child.global_start_time == 0.35

    result = s.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'V'
    expected = np.zeros(N_PTS)
    expected[350:400] = 2.0
    assert np.array_equal(result.data, expected)

    parent = stimuli.Stimulus(description="outer")
    s2 = stimuli.Stimulus(description="inner", parent=parent)
    assert s2.parent is parent
    assert parent.items == (s2,)

    assert np.array_equal(stimuli.Stimulus(units='A').eval(t0=0, n_pts=10, dt=DT).data, np.zeros(10))
    _check_round_trip(s)


def test_offset_params():
    stim = stimuli.Offset(amplitude=-1.5, start_time=0.3, description="holding", units='A')
    assert stim.description == "holding"
    assert stim.units == 'A'
    result = stim.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'A'
    expected = np.zeros(N_PTS)
    expected[300:] = -1.5
    assert np.array_equal(result.data, expected)

    parent = stimuli.Stimulus(start_time=0.2, items=[stim])
    expected = np.zeros(N_PTS)
    expected[500:] = -1.5
    assert np.array_equal(_eval(parent), expected)
    assert np.array_equal(stim.mask(t0=0, n_pts=N_PTS, dt=DT).data, expected != 0)

    loaded = _check_round_trip(stimuli.Offset(amplitude=-1.5, start_time=0.3, description="holding", units='A'))
    assert loaded.amplitude == -1.5


def test_square_pulse_params():
    stim = stimuli.SquarePulse(start_time=0.05, duration=0.1, amplitude=2e-10, description="test pulse", units='A')
    result = stim.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'A'
    expected = np.zeros(N_PTS)
    expected[50:150] = 2e-10
    assert np.array_equal(result.data, expected)
    loaded = _check_round_trip(stim)
    assert (loaded.description, loaded.units, loaded.duration, loaded.amplitude) == ("test pulse", 'A', 0.1, 2e-10)

    # zero-duration pulse is allowed and affects no samples
    assert np.array_equal(_eval(stimuli.SquarePulse(start_time=0.05, duration=0, amplitude=1)), np.zeros(N_PTS))


def test_square_pulse_train_params():
    stim = stimuli.SquarePulseTrain(
        start_time=0.1, n_pulses=4, pulse_duration=0.005, amplitude=-3.0, interval=0.02,
        description="train", units='V',
    )
    assert len(stim.items) == 4
    assert np.allclose(stim.pulse_times, [0, 0.02, 0.04, 0.06])
    assert np.allclose(stim.global_pulse_times, [0.1, 0.12, 0.14, 0.16])
    for i, pulse in enumerate(stim.items):
        assert pulse.pulse_number == i
        assert pulse.units == 'V'
        assert pulse.amplitude == -3.0
        assert pulse.duration == 0.005

    expected = np.zeros(N_PTS)
    for start in (100, 120, 140, 160):
        expected[start:start + 5] = -3.0
    assert np.array_equal(_eval(stim), expected)

    assert stim.save()['items'] == []
    loaded = _check_round_trip(stim)
    assert len(loaded.items) == 4
    assert (loaded.description, loaded.units, loaded.n_pulses, loaded.interval) == ("train", 'V', 4, 0.02)


def test_square_pulse_series_params():
    stim = stimuli.SquarePulseSeries(
        start_time=0.2, pulse_times=[0, 0.05, 0.12], pulse_durations=[0.01, 0.02, 0.005],
        amplitudes=[1.0, -2.0, 0.5], description="series", units='A',
    )
    assert len(stim.items) == 3
    assert np.allclose(stim.global_pulse_times, [0.2, 0.25, 0.32])
    expected = np.zeros(N_PTS)
    expected[200:210] = 1.0
    expected[250:270] = -2.0
    expected[320:325] = 0.5
    assert np.array_equal(_eval(stim), expected)
    assert stim.save()['items'] == []
    loaded = _check_round_trip(stim)
    assert loaded.description == "series"
    assert len(loaded.items) == 3

    default = stimuli.SquarePulseSeries(start_time=0, pulse_times=[0], pulse_durations=[0.01], amplitudes=[1])
    assert default.description == "square pulse series"

    # numpy array / scalar inputs must save as JSON-compatible values and compare equal after load
    np_stim = stimuli.SquarePulseSeries(
        start_time=np.float64(0.2), pulse_times=np.array([0, 0.05, 0.12]),
        pulse_durations=np.array([0.01, 0.02, 0.005]), amplitudes=np.array([1.0, -2.0, 0.5], dtype=np.float32),
    )
    assert np_stim == stimuli.SquarePulseSeries(
        start_time=0.2, pulse_times=[0, 0.05, 0.12], pulse_durations=[0.01, 0.02, 0.005],
        amplitudes=[1.0, -2.0, 0.5],
    )
    _check_round_trip(np_stim)


def test_ramp_params():
    stim = stimuli.Ramp(start_time=0.1, duration=0.5, slope=4.0, offset=-1.0, description="ramp", units='V')
    result = stim.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'V'
    expected = np.zeros(N_PTS)
    expected[100:600] = -1.0 + 4.0 * np.arange(500) * DT
    assert np.allclose(result.data, expected, rtol=0, atol=1e-12)
    assert result.data[99] == 0
    assert result.data[600] == 0

    # slope is per second, so the value at a given time is independent of sample rate
    for dt in (1e-3, 1e-4, 2e-5):
        trace = stim.eval(t0=0, n_pts=int(round(1.0 / dt)), dt=dt)
        assert np.isclose(trace.value_at(0.35, interp='nearest'), 0.0, atol=4.0 * dt)
        assert np.isclose(trace.value_at(0.6 - dt, interp='nearest'), -1.0 + 4.0 * (0.5 - dt), atol=1e-9)

    state = stim.save()
    assert state['args']['offset'] == -1.0
    assert state['args']['slope'] == 4.0
    loaded = _check_round_trip(stim)
    assert loaded.offset == -1.0


def test_sine_params():
    stim = stimuli.Sine(
        start_time=0.1, duration=0.5, frequency=10.0, amplitude=2.0, phase=np.pi / 2, offset=0.5,
        description="sine", units='A',
    )
    result = stim.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'A'
    t = np.arange(500) * DT
    expected = np.zeros(N_PTS)
    expected[100:600] = 0.5 + 2.0 * np.cos(2 * np.pi * 10.0 * t)
    assert np.allclose(result.data, expected, rtol=0, atol=1e-12)
    assert np.isclose(result.data[100], 2.5)   # phase pi/2 -> peak at start
    assert np.isclose(result.data[125], 0.5)   # quarter period later -> zero crossing
    assert np.isclose(result.data[150], -1.5)  # half period later -> trough
    assert result.data[99] == 0
    assert result.data[600] == 0

    assert stim.save()['args']['offset'] == 0.5
    loaded = _check_round_trip(stim)
    assert loaded.offset == 0.5
    assert loaded.phase == np.pi / 2


def test_chirp_params():
    f0, f1, dur, amp, ph, off = 5.0, 40.0, 1.0, 1.5, 0.3, -0.2
    stim = stimuli.Chirp(
        start_time=0.1, duration=dur, start_frequency=f0, end_frequency=f1, amplitude=amp,
        phase=ph, offset=off, description="chirp", units='V',
    )
    result = stim.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'V'
    # phase = ph + integral of 2*pi*f(t) with f(t) = f0 * (f1/f0)**(t/dur)
    t = np.arange(1000) * DT
    r = f1 / f0
    phase = ph + 2 * np.pi * f0 * dur * (r ** (t / dur) - 1) / np.log(r)
    expected = np.zeros(N_PTS)
    expected[100:1100] = off + amp * np.sin(phase)
    assert np.allclose(result.data, expected, rtol=0, atol=1e-9)
    assert result.data[99] == 0
    assert result.data[1100] == 0
    assert np.isclose(result.data[100], off + amp * np.sin(ph))

    assert np.isclose(stim.frequency_at(0), f0)
    assert np.isclose(stim.frequency_at(dur), f1)
    assert np.isclose(stim.frequency_at(dur / 2), np.sqrt(f0 * f1))
    # frequency_at agrees with the numerical derivative of phase_at
    tt = np.linspace(0.01, 0.99, 20)
    h = 1e-6
    numeric_freq = (stim.phase_at(tt + h) - stim.phase_at(tt - h)) / (2 * h) / (2 * np.pi)
    assert np.allclose(numeric_freq, stim.frequency_at(tt), rtol=1e-5)

    loaded = _check_round_trip(stim)
    assert (loaded.phase, loaded.offset, loaded.start_frequency, loaded.end_frequency) == (ph, off, f0, f1)


def test_chirp_constant_frequency():
    chirp = stimuli.Chirp(start_time=0.1, duration=0.5, start_frequency=8.0, end_frequency=8.0, amplitude=1.2, phase=0.4, offset=0.1)
    sine = stimuli.Sine(start_time=0.1, duration=0.5, frequency=8.0, amplitude=1.2, phase=0.4, offset=0.1)
    data = _eval(chirp)
    assert np.all(np.isfinite(data))
    assert np.allclose(data, _eval(sine), rtol=0, atol=1e-12)
    assert np.isclose(chirp.frequency_at(0.25), 8.0)
    assert np.allclose(chirp.frequency_at(np.array([0, 0.5])), [8.0, 8.0])


def test_psp_params():
    dt = 1e-4
    n_pts = 5000
    rise_time, decay_tau, amp = 0.004, 0.02, -3e-3
    stim = stimuli.Psp(start_time=0.05, rise_time=rise_time, decay_tau=decay_tau, amplitude=amp, rise_power=3, description="epsc", units='A')
    assert stim.duration == 15 * decay_tau
    result = stim.eval(t0=0, n_pts=n_pts, dt=dt)
    assert result.units == 'A'
    data = result.data
    start_ind = 500
    peak_ind = start_ind + int(round(rise_time / dt))
    end_ind = start_ind + int(round(stim.duration / dt))

    # zero before onset and after the stimulus duration
    assert np.all(data[:start_ind + 1] == 0)
    assert np.all(data[end_ind:] == 0)
    assert data[end_ind - 1] != 0

    # peak of the given amplitude at rise_time after onset
    assert np.argmin(data) == peak_ind
    assert np.isclose(data[peak_ind], amp, rtol=1e-6)

    # late decay follows exp(-t / decay_tau)
    i1 = start_ind + int(round(0.1 / dt))
    i2 = start_ind + int(round(0.15 / dt))
    assert np.isclose(data[i2] / data[i1], np.exp(-0.05 / decay_tau), rtol=1e-6)

    assert stimuli.Psp(start_time=0, rise_time=0.001, decay_tau=0.01, amplitude=1).description == "PSP"
    loaded = _check_round_trip(stim)
    assert (loaded.rise_time, loaded.decay_tau, loaded.amplitude, loaded.rise_power) == (rise_time, decay_tau, amp, 3)


def test_lazy_load_stimulus():
    loader = FakeLoader({
        'rec1': [
            stimuli.SquarePulse(start_time=0.1, duration=0.05, amplitude=2.0, units='A'),
            stimuli.Offset(amplitude=0.5, start_time=0.3, units='A'),
        ],
    })
    stim = stimuli.LazyLoadStimulus(description="lazy", start_time=0.2, units='A', loader=loader, source='rec1')
    assert loader.calls == []
    assert stim.description == "lazy"
    assert stim.global_start_time == 0.2

    items = stim.items
    assert loader.calls == ['rec1']
    assert len(items) == 2
    assert all(item.parent is stim for item in items)
    stim.items
    assert loader.calls == ['rec1']

    result = stim.eval(t0=0, n_pts=N_PTS, dt=DT)
    assert result.units == 'A'
    expected = np.zeros(N_PTS)
    expected[300:350] = 2.0
    expected[500:] += 0.5
    assert np.array_equal(result.data, expected)

    # loader is not serializable; saved state is a plain Stimulus holding the loaded items
    state = stim.save()
    json_state = json.loads(json.dumps(state))
    assert json_state == state
    assert state['type'] == 'Stimulus'
    for st in (state, json_state):
        loaded = stimuli.Stimulus.load(st)
        assert type(loaded) is stimuli.Stimulus
        assert loaded.items == stim.items
        assert (loaded.description, loaded.start_time, loaded.units) == ("lazy", 0.2, 'A')
        assert np.array_equal(_eval(loaded), expected)

    with pytest.raises(Exception):
        stimuli.LazyLoadStimulus(description="no loader", source='rec1')
    with pytest.raises(Exception):
        stimuli.LazyLoadStimulus(description="no source", loader=loader)


# --- hierarchy, loading, and equality tests ---

def _build_hierarchy():
    root = stimuli.Stimulus(description="root", start_time=0.05, units='A')
    stimuli.SquarePulse(start_time=0.01, duration=0.02, amplitude=-1.0, parent=root, units='A')
    stimuli.SquarePulseTrain(start_time=0.1, n_pulses=3, pulse_duration=0.004, amplitude=2.0, interval=0.01, parent=root, units='A')
    stimuli.Ramp(start_time=0.2, duration=0.1, slope=10.0, offset=0.25, parent=root, units='A')
    stimuli.Sine(start_time=0.35, duration=0.1, frequency=20.0, amplitude=0.3, phase=1.0, offset=0.1, parent=root, units='A')
    stimuli.Chirp(start_time=0.5, duration=0.3, start_frequency=5.0, end_frequency=50.0, amplitude=0.4, phase=0.2, offset=-0.1, parent=root, units='A')
    stimuli.Psp(start_time=0.85, rise_time=0.003, decay_tau=0.01, amplitude=0.7, rise_power=2.5, parent=root, units='A')
    stimuli.Offset(amplitude=0.05, start_time=1.5, parent=root, units='A')
    nested = stimuli.Stimulus(description="nested", start_time=1.1, parent=root, units='A')
    stimuli.SquarePulseSeries(start_time=0.02, pulse_times=[0, 0.03], pulse_durations=[0.01, 0.015], amplitudes=[0.6, -0.6], parent=nested, units='A')
    stimuli.SquarePulse(start_time=0.1, duration=0.05, amplitude=3.0, parent=nested, units='A')
    return root


def test_nested_hierarchy():
    root = _build_hierarchy()
    nested = root.items[-1]
    assert len(root.items) == 8
    assert nested.items[1].global_start_time == pytest.approx(0.05 + 1.1 + 0.1)

    data = _eval(root)
    assert np.allclose(data, sum(_eval(item) for item in root.items))
    # nested pulse at global 1.25 s (index 1250) for 0.05 s, with nothing else active
    assert np.all(data[1250:1300] == 3.0)
    # first square pulse at global 0.06 s for 0.02 s
    assert np.all(data[60:80] == -1.0)

    loaded = _check_round_trip(root)
    assert loaded.items[-1].items[1].global_start_time == pytest.approx(1.25)
    assert loaded.items[-1].items[0].items[1].global_start_time == pytest.approx(1.2)

    # a differing grandchild makes the hierarchies unequal
    other = _build_hierarchy()
    other.items[-1].items[1].amplitude = 3.5
    assert other != root


def test_hierarchy_durations():
    parent = stimuli.Stimulus(start_time=1.0)
    stimuli.SquarePulse(start_time=0.5, duration=0.25, parent=parent, amplitude=1)
    assert parent.global_end_time == 1.0
    assert parent.total_global_end_time == 1.75
    assert parent.total_duration == 0.75


def test_load_sub_subclass():
    stim = CustomPulseVariant(start_time=0.1, duration=0.2, amplitude=1.5, units='A')
    loaded = _check_round_trip(stim)
    assert type(loaded) is CustomPulseVariant
    assert type(stimuli.SquarePulse.load(stim.save())) is CustomPulseVariant
    assert stimuli.Stimulus.get_stimulus_class('CustomPulse') is CustomPulse

    # loading through a subclass still resolves classes outside that subclass's tree
    parent = stimuli.Stimulus(items=[stimuli.Ramp(start_time=0, duration=0.1, slope=1)])
    reloaded = CustomPulse.load(parent.save())
    assert type(reloaded) is stimuli.Stimulus
    assert reloaded == parent

    # a class defined after the registry was populated is still found
    stimuli.Stimulus.get_stimulus_class('Stimulus')

    class LateDefinedPulse(CustomPulseVariant):
        pass

    late = LateDefinedPulse(start_time=0, duration=0.1, amplitude=1)
    assert type(stimuli.Stimulus.load(late.save())) is LateDefinedPulse

    with pytest.raises(KeyError):
        stimuli.Stimulus.get_stimulus_class('NoSuchStimulus')


def test_load_items_on_non_container_class():
    pulse = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=1.0)
    stimuli.Offset(amplitude=0.5, start_time=0.1, parent=pulse)
    loaded = _check_round_trip(pulse)
    assert len(loaded.items) == 1
    assert loaded.items[0].parent is loaded


def test_equality():
    a = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=1.0)
    b = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=1.0)
    assert a == b
    assert not (a != b)
    assert a != stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=2.0)
    assert a != stimuli.Ramp(start_time=0.1, duration=0.2, slope=1.0)
    assert a != None
    assert a != 1
    assert not (a == "square pulse")

    b.append_item(stimuli.Offset(amplitude=1, start_time=0))
    assert a != b


def test_identical_siblings():
    parent = stimuli.Stimulus()
    a = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=1.0, parent=parent)
    b = stimuli.SquarePulse(start_time=0.1, duration=0.2, amplitude=1.0, parent=parent)
    assert a == b
    assert len(parent.items) == 2
    assert parent.items[0] is a and parent.items[1] is b

    parent.remove_item(b)
    assert len(parent.items) == 1
    assert parent.items[0] is a
    assert b.parent is None and a.parent is parent

    b.parent = parent
    a.parent = None
    assert len(parent.items) == 1
    assert parent.items[0] is b

    with pytest.raises(ValueError):
        parent.remove_item(a)


def test_save_numpy_types():
    stim = stimuli.SquarePulse(start_time=np.float64(0.1), duration=np.float32(0.25), amplitude=np.int64(3), units=np.str_('A'))
    args = stim.save()['args']
    assert type(args['start_time']) is float
    assert type(args['duration']) is float
    assert type(args['amplitude']) is int
    assert type(args['units']) is str
    _check_round_trip(stim)
