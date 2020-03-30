import numpy as np
from neuroanalysis.data.dataset import Recording, TSeries
import neuroanalysis.analyzers.stim_pulse as spas
from pytest import raises


def test_generic_stim_pulse_analyzer():

    ### test a recording with two pulses and noisy data
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

    rec1 = Recording(
        channels = {'reporter':TSeries(data=data, dt=dt, units='V')},
        device_id='test',
        start_time = 0)

    spa1 = spas.GenericStimPulseAnalyzer.get(rec1)

    with raises(ValueError):
        spa1.pulses()

    pulses = spa1.pulses(channel='reporter')
    assert len(pulses) == 2
    assert np.isclose(pulses[0][0], 0.4)
    assert np.isclose(pulses[0][2], amp1, 0.001, 0.001)
    assert np.isclose(pulses[0][1], 0.44)
    assert np.isclose(pulses[1][0], 1)
    assert np.isclose(pulses[1][2], amp2, 0.001, 0.001)
    assert np.isclose(pulses[1][1], 1.1)


    ### test a noise-free recording with 3 pulses
    data = np.zeros(10000)
    data[5000:5020] = 1
    data[6000:6020] = 1
    data[7000:7020] = 1

    rec2 = Recording(
        channels = {'reporter':TSeries(data=data, dt=dt, units='V')},
        device_id='test',
        start_time = 0)

    spa2 = spas.GenericStimPulseAnalyzer.get(rec2)

    pulses = spa2.pulses(channel='reporter')

    ## check values of pulses found
    assert len(pulses) == 3
    for p in pulses:
        assert isinstance(p, tuple)
        assert np.isclose(p[2], 1)
        assert np.isclose(p[1]-p[0], 0.004)
    assert np.isclose(pulses[0][0], 1)
    assert np.isclose(pulses[1][0], 1.2)
    assert np.isclose(pulses[2][0], 1.4)

    ### check stim params
    freq, delay = spa2.stim_params(channel='reporter')
    assert np.isclose(freq, 5.0)
    assert np.isclose(delay, 0.2)




