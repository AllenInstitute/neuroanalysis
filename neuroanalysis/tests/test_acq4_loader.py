import numpy as np
import pytest

pytest.importorskip('acq4.util.DataManager')
from acq4.util import DataManager
from MetaArray import MetaArray, axis

import neuroanalysis.stimuli as stimuli
from neuroanalysis.data.dataset import Dataset, PatchClampRecording, Recording
from neuroanalysis.data.loaders.acq4_dataset_loader import (
    Acq4DatasetLoader, legacy_stimulus, recorded_stimulus, trial_sequence_values,
)


def seq_param(value, **seq):
    return {'sequence': 'off', **seq, 'value': value}


def legacy_pulse(start, length, amplitude, sequences=None):
    item = {'start': seq_param(start), 'length': seq_param(length), 'amplitude': seq_param(amplitude),
            'sum': {'sequence': 'off', 'affect': 'length', 'value': length * amplitude}, 'type': 'pulse'}
    for k, v in (sequences or {}).items():
        item[k].update(v)
    return item


def legacy_train(start, length, amplitude, pulse_number, period=None, interpulse_length=None):
    item = legacy_pulse(start, length, amplitude)
    item['type'] = 'pulseTrain'
    if period is not None:
        item['period'] = seq_param(period)
    if interpulse_length is not None:
        item['interpulse_length'] = seq_param(interpulse_length)
    item['pulse_number'] = seq_param(pulse_number)
    return item


def generator_state(stimuli_state, advanced=False):
    state = {'function': 'pulse(...)', 'params': {}, 'autoUpdate': True, 'advancedMode': advanced}
    if stimuli_state is not None:
        state['stimuli'] = stimuli_state
    return state


def multiclamp_state(stimuli_state, mode='VC', holding=None, advanced=False):
    return {'mode': mode, 'holdingCheck': holding is not None, 'holdingSpin': holding or 0.0,
            'waveGeneratorWidget': generator_state(stimuli_state, advanced)}


def daq_clamp_state(stimuli_state, mode='IC', holding=None):
    command = {'holdingCheck': holding is not None, 'holdingSpin': holding or 0.0, 'preSetCheck': False,
               'waveGeneratorWidget': generator_state(stimuli_state)}
    return {'mode': mode, 'daqState': {'channels': {'command': command, 'primary': {'recordCheck': True}}}}


def stimgen_stimulus():
    return stimuli.Stimulus('StimGenerator', units='V', items=[
        stimuli.Offset(-0.07),
        stimuli.SquarePulse(0.01, 0.02, 0.01),
        stimuli.SquarePulseTrain(0.05, 3, 0.002, 0.005, 0.01),
    ])


def test_recorded_stimulus_clamp():
    stim = stimgen_stimulus()
    loaded = recorded_stimulus({'mode': 'VC', 'stimulus': stim.save()}, channel='command')
    assert loaded == stim
    assert [type(i) for i in loaded.items] == [stimuli.Offset, stimuli.SquarePulse, stimuli.SquarePulseTrain]
    assert loaded.units == 'V'


def test_recorded_stimulus_daq_channel():
    stim = stimgen_stimulus()
    protocol = {'command': {'holding': 0, 'stimulus': stim.save()}, 'primary': {'record': True}}
    assert recorded_stimulus(protocol, channel='command') == stim
    assert recorded_stimulus(protocol) == stim
    assert recorded_stimulus(protocol, channel='primary') is None
    assert recorded_stimulus({'command': {'holding': 0}, 'primary': {'record': True}}) is None


def test_recorded_stimulus_ambiguous_channels():
    stim = stimgen_stimulus().save()
    with pytest.raises(NotImplementedError):
        recorded_stimulus({'power': {'stimulus': stim}, 'shutter': {'stimulus': stim}})


def test_legacy_pulse():
    state = multiclamp_state({'Pulse': legacy_pulse(0.01, 0.02, -0.01)})
    stim = legacy_stimulus(state, description='IV', units='V')
    assert stim.description == 'IV'
    assert stim.units == 'V'
    assert len(stim.items) == 1
    pulse = stim.items[0]
    assert isinstance(pulse, stimuli.SquarePulse)
    assert (pulse.start_time, pulse.duration, pulse.amplitude) == (0.01, 0.02, -0.01)
    assert pulse.description == 'Pulse'
    assert pulse.units == 'V'


def test_legacy_pulse_train_period_and_interpulse_length():
    state = multiclamp_state({
        'PulseTrain': legacy_train(0.1, 0.002, 1e-10, 5, period=0.02),
        'PulseTrain2': legacy_train(0.3, 0.001, 2e-10, 3, interpulse_length=0.01),
    }, mode='IC')
    stim = legacy_stimulus(state, units='A')
    train1, train2 = stim.items
    assert isinstance(train1, stimuli.SquarePulseTrain)
    assert (train1.start_time, train1.n_pulses, train1.pulse_duration, train1.amplitude, train1.interval) == (
        0.1, 5, 0.002, 1e-10, 0.02)
    assert (train2.start_time, train2.n_pulses, train2.pulse_duration, train2.amplitude, train2.interval) == (
        0.3, 3, 0.001, 2e-10, 0.01)
    assert np.allclose(train2.global_pulse_times, [0.3, 0.31, 0.32])


def test_legacy_sequence_values():
    state = multiclamp_state({
        'Pulse': legacy_pulse(0.01, 0.02, -0.01, {'amplitude': {'sequence': 'list', 'list': '-10mV, 10mV'}}),
        'PulseTrain': legacy_train(0.1, 0.002, 0.01, 5, period=0.02),
    })
    stim = legacy_stimulus(state, {'Pulse_amplitude': 0.01, 'PulseTrain_pulse_number': 7.0, 'PulseTrain_period': 0.03})
    pulse, train = stim.items
    assert pulse.amplitude == 0.01
    assert train.n_pulses == 7 and train.interval == 0.03

    # unsequenced params, or sequenced params with no trial value, keep their default
    pulse, train = legacy_stimulus(state).items
    assert pulse.amplitude == -0.01
    assert train.n_pulses == 5


def test_legacy_sequenced_sum():
    item = legacy_pulse(0.01, 0.02, 0.5)
    state = multiclamp_state({'Pulse': item})
    assert legacy_stimulus(state, {'Pulse_sum': 0.02}).items[0].duration == pytest.approx(0.04)
    item['sum']['affect'] = 'amplitude'
    assert legacy_stimulus(state, {'Pulse_sum': 0.02}).items[0].amplitude == pytest.approx(1.0)


@pytest.mark.parametrize('holding', [None, -0.065])
def test_legacy_holding_multiclamp(holding):
    state = multiclamp_state({'Pulse': legacy_pulse(0.01, 0.02, -0.01)}, holding=holding)
    stim = legacy_stimulus(state, units='V')
    if holding is None:
        assert [type(i) for i in stim.items] == [stimuli.SquarePulse]
    else:
        offset = stim.items[0]
        assert isinstance(offset, stimuli.Offset)
        assert offset.amplitude == holding
        assert offset.units == 'V'
        assert stim.eval(n_pts=100, sample_rate=1000).data[0] == holding


def test_legacy_daq_clamp_state():
    state = daq_clamp_state({'Pulse': legacy_pulse(0.01, 0.02, 1e-10)}, holding=5e-11)
    stim = legacy_stimulus(state, {'command.Pulse_amplitude': 2e-10}, units='A')
    offset, pulse = stim.items
    assert offset.amplitude == 5e-11 and offset.units == 'A'
    assert pulse.amplitude == 2e-10


def test_legacy_advanced_mode():
    with pytest.raises(NotImplementedError):
        legacy_stimulus(multiclamp_state(None, advanced=True))
    # pre-simple-mode states have only a function string
    with pytest.raises(NotImplementedError):
        legacy_stimulus(multiclamp_state(None))
    with pytest.raises(NotImplementedError):
        legacy_stimulus(multiclamp_state({'format': 2, 'items': []}))


def test_legacy_no_generator():
    assert legacy_stimulus({'mode': 'VC', 'holdingCheck': False}) is None


def test_trial_sequence_values():
    seq_info = {'sequenceParams': {('Clamp1', 'Pulse_amplitude'): [-0.01, 0.0, 0.01],
                                   ('Clamp1', 'Pulse_length'): [0.01, 0.02],
                                   ('Clamp2', 'Pulse_amplitude'): [1, 2]}}
    trial_info = {('Clamp1', 'Pulse_amplitude'): 2, ('Clamp2', 'Pulse_amplitude'): 0, 'dirType': 'Protocol'}
    assert trial_sequence_values(seq_info, trial_info, 'Clamp1') == {'Pulse_amplitude': 0.01}


# -- end-to-end through DataManager on a temporary directory --

def write_clamp_file(dh, name, protocol, mode='VC', holding=-0.065):
    units = ('A', 'V') if mode == 'VC' else ('V', 'A')
    info = [
        axis(name='Channel', cols=[('primary', units[0]), ('command', units[1])]),
        axis(name='Time', units='s', values=np.arange(100) * 1e-4),
        {'ClampState': {'mode': mode, 'holding': holding}, 'Protocol': protocol},
    ]
    return dh.writeFile(MetaArray(np.zeros((2, 100)), info=info), name + '.ma')


def make_sequence(tmp_path, dev_state, protocol, seq_params, device='Clamp1'):
    """Write a ProtocolSequence dir with one trial dir per index in seq_params' value lists."""
    root = DataManager.getDirHandle(str(tmp_path))
    seq = root.mkdir('IV_000', info={'dirType': 'ProtocolSequence', 'devices': {device: dev_state},
                                     'sequenceParams': {(device, k): v for k, v in seq_params.items()}})
    n = max([len(v) for v in seq_params.values()] or [1])
    for i in range(n):
        trial = seq.mkdir('%03d' % i, info={'dirType': 'Protocol', 'startTime': 1e9,
                                            **{(device, k): i for k in seq_params}})
        write_clamp_file(trial, device, protocol)
    return root


def clamp_recording(fh, device='Clamp1', clamp_mode='vc'):
    return PatchClampRecording(channels={}, device_id=device, file_name=fh.name(), clamp_mode=clamp_mode,
                               loader=Acq4DatasetLoader(fh.parent().parent().name()))


def test_load_stimulus_recorded(tmp_path):
    stim = stimgen_stimulus()
    root = make_sequence(tmp_path, multiclamp_state({'Pulse': legacy_pulse(0.01, 0.02, -0.01)}),
                         {'mode': 'VC', 'stimulus': stim.save()}, {})
    rec = clamp_recording(root['IV_000']['000']['Clamp1.ma'])
    assert rec.stimulus == stim


def test_load_stimulus_recorded_daq_clamp(tmp_path):
    stim = stimgen_stimulus()
    protocol = {'command': {'stimulus': stim.save()}, 'primary': {'record': True}}
    root = make_sequence(tmp_path, daq_clamp_state(None), protocol, {}, device='AxoPatch200')
    rec = clamp_recording(root['IV_000']['000']['AxoPatch200.ma'], device='AxoPatch200')
    assert rec.stimulus == stim


def test_load_stimulus_legacy_sequence(tmp_path):
    state = multiclamp_state({'Pulse': legacy_pulse(0.01, 0.02, -0.01, {'amplitude': {'sequence': 'list'}})},
                             holding=-0.065)
    root = make_sequence(tmp_path, state, {'mode': 'VC'}, {'Pulse_amplitude': [-0.02, 0.0, 0.02]})
    loader = Acq4DatasetLoader(root['IV_000'].name())
    amps = []
    for i in range(3):
        rec = clamp_recording(root['IV_000']['%03d' % i]['Clamp1.ma'])
        stim = loader.load_stimulus(rec)
        assert stim.description == 'IV'
        assert stim.units == 'V'
        offset, pulse = stim.items
        assert offset.amplitude == -0.065
        amps.append(pulse.amplitude)
    assert amps == [-0.02, 0.0, 0.02]


def test_load_stimulus_through_dataset(tmp_path):
    state = multiclamp_state({'Pulse': legacy_pulse(0.01, 0.02, -0.01, {'length': {'sequence': 'list'}})})
    root = make_sequence(tmp_path, state, {'mode': 'VC'}, {'Pulse_length': [0.01, 0.03]})
    ds = Dataset(loader=Acq4DatasetLoader(root.name()))
    durations = []
    for srec in ds.contents:
        rec = srec.recordings[0]
        assert isinstance(rec, PatchClampRecording)
        durations.append(rec.stimulus.items[0].duration)
    assert durations == [0.01, 0.03]


def test_load_stimulus_advanced_mode(tmp_path):
    root = make_sequence(tmp_path, multiclamp_state(None, advanced=True), {'mode': 'VC'}, {})
    rec = clamp_recording(root['IV_000']['000']['Clamp1.ma'])
    with pytest.raises(NotImplementedError):
        rec.loader.load_stimulus(rec)


def test_load_stimulus_generic_recording():
    stim = stimgen_stimulus()
    rec = Recording(channels={}, device_type='Laser', Protocol={'power': {'stimulus': stim.save()}, 'shutter': {}})
    assert Acq4DatasetLoader('unused').load_stimulus(rec) == stim
    with pytest.raises(NotImplementedError):
        Acq4DatasetLoader('unused').load_stimulus(Recording(channels={}, device_type='Laser', Protocol={}))
