from collections import OrderedDict

import numpy as np

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
    test_data[100:250] = 0.5 + np.arange(150) * 2.3
    assert np.all(t.data == test_data)
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












