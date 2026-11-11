from datetime import datetime
from neuroanalysis.data.loaders.loaders import DatasetLoader
from acq4.util import DataManager
from acq4.analysis.dataModels import PatchEPhys
from neuroanalysis.data.dataset import Recording, SyncRecording, TSeries, PatchClampRecording
import neuroanalysis.stimuli as stimuli


class Acq4DatasetLoader(DatasetLoader):
    def __init__(self, filepath):

        self._filepath = filepath

        self._dh = None ## directory handle

    @property
    def dh(self):
        if self._dh is None:
            self._dh = DataManager.getDirHandle(self._filepath)
        return self._dh

    def get_dataset_name(self):
        return self._filepath

    def get_sync_recordings(self, dataset):
        """Return a list of SyncRecordings."""

        sync_recs = []
        sequences = []

        for seq in self.dh.subDirs():
            if self.dh[seq].info().get('dirType') == 'ProtocolSequence':
                params = self.dh[seq].info()['sequenceParams']   
                #sequence = RecordingSequence(parent=dataset, name=seq, meta={'sequence_params':params}, loader=self) 
                for sd in self.dh[seq].subDirs():
                    sdh = self.dh[seq][sd]
                    meta = {'sequence_params':{}}
                    for k in params.keys():
                        meta['sequence_params'][k] = params[k][sdh.info().get(k)]
                    srec = SyncRecording(parent=dataset, key=(seq,sd), meta=meta, loader=self)
                    #sequence.add_sync_rec(srec)
                    sync_recs.append(srec)
                #sequences.append(sequence)
            elif self.dh[seq].info().get('dirType') == 'Protocol':
                srec = SyncRecording(parent=dataset, key=(seq), loader=self)
                sync_recs.append(srec)
            elif self.dh[seq].shortName() == 'Patch': ## ignore this for now -- how should this data be represented?
                continue
            else:
                raise ValueError(f'Not sure how to handle folder {self.dh[seq].name()}')

        #return (sync_recs, sequences)
        return sync_recs
        
    def get_recordings(self, sync_rec):
        """Return a dict of {device: Recording}"""
        key = sync_rec.key
        dh = self.dh
        for k in key:
            dh = dh[k]

        ## build a flat list of all the files in dh
        ls = [dh[f] for f in dh.ls()]
        files = []
        for f in ls:
            if f.isDir():
                ls.extend([f[x] for x in f.ls()]) # -- broken - need to deal with str - fh conversion :/
            else:
                files.append(f)

        recordings = {}
        for f in files:
            start_time=datetime.utcfromtimestamp(PatchEPhys.getParent(f, 'Protocol').info()['startTime'])

            if PatchEPhys.isClampFile(f):
                meta = {
                    'file_name': f.name(),
                    'clamp_mode': PatchEPhys.getClampMode(f).lower(),
                }
                if meta['clamp_mode'] == 'vc':
                    meta['holding_current'] = PatchEPhys.getClampHoldingLevel(f)
                elif meta['clamp_mode'] == 'ic':
                    meta['holding_potential'] = PatchEPhys.getClampHoldingLevel(f)
                    meta['bridge_balance'] = PatchEPhys.getBridgeBalanceCompensation(f)
                else:
                    raise ValueError(f"dont know how to interpret {meta['clamp_mode']} clamp_mode")

                data = f.read()
                dt = data.axisValues(1)[1] - data.axisValues(1)[0]

                rec = PatchClampRecording(
                    channels={'primary':TSeries(channel_id='primary', data=data['primary'].asarray(), dt=dt, units=data.columnUnits(0, 'primary'), loader=self),
                              'command':TSeries(channel_id='command', data=data['command'].asarray(), dt=dt, units=data.columnUnits(0, 'command'), loader=self)},
                    start_time=start_time, 
                    device_type='patch clamp amplifier', 
                    device_id=f.shortName().strip('.ma'),
                    sync_recording=sync_rec,
                    loader=self,
                    **meta
                    )
                rec['primary']._recording = rec
                rec['command']._recording = rec

            else:
                data = f.read()
                time_axis = data._getAxis('Time') ## is there a way to do this without using a private MA function?
                if 'Channel' in data.listColumns().keys():
                    channel_axis = data._getAxis('Channel')

                #dt = data.axisValues(time_axis)[1] - data.axisValues(time_axis)[0]

                if data.ndim == 2:
                    channels = {k:TSeries(channel_id=k, data=data[k].asarray(), time_values=data.axisValues(time_axis), units=data.columnUnits(channel_axis, k), loader=self) for k in data.listColumns(channel_axis)}
                elif data.ndim == 3:
                    channels = {'frames':TSeries(channel_id='frames', data=data.asarray(), time_values=data.axisValues(time_axis), loader=self)}

                rec = Recording(
                    channels=channels,
                    start_time=start_time,
                    device_type=f.name(relativeTo=dh).strip('.ma'),
                    sync_recording=sync_rec,
                    file_name=f.name(),
                    loader=self,
                    **data.infoCopy()[-1]
                    )

                for k in rec.channels:
                    rec[k]._recording = rec

            recordings[f.name(relativeTo=dh)] = rec

        return recordings

    def get_tseries_data(self, tseries):
        """Return a numpy array of the data in the tseries."""
        #### I don't think we need this because we hand TSeries their data when we instantiate them.
        raise NotImplementedError("Must be implemented in subclass. -- This should only get called if we're using lazy loading.")

    def load_stimulus(self, recording) -> stimuli.Stimulus | None:
        """Return the Stimulus applied during *recording*.

        Uses the stimulus saved in the recording's ``.ma`` Protocol info. Clamp recordings without
        one fall back to the legacy simple-mode stimulus generator state (see `legacy_stimulus`).
        Returns None if the task has no stimulus generator state.
        Raises NotImplementedError for stimuli that cannot be reconstructed.
        """
        if not isinstance(recording, PatchClampRecording):
            stim = recorded_stimulus(recording.meta.get('Protocol', {}))
            if stim is None:
                raise NotImplementedError('Recording has no saved stimulus; legacy stimuli are only parsed for clamp recordings.')
            return stim

        fh = DataManager.getFileHandle(recording.meta['file_name'])
        info = fh.read(readAllData=False).infoCopy()[-1]
        stim = recorded_stimulus(info.get('Protocol', {}), channel='command')
        if stim is not None:
            return stim

        trial_dir = PatchEPhys.getParent(fh, 'Protocol')
        seq_dir = PatchEPhys.getParent(fh, 'ProtocolSequence')
        task_dir = trial_dir if seq_dir is None else seq_dir
        if task_dir is None:
            return None
        dev_state = task_dir.info().get('devices', {}).get(recording.device_id)
        if dev_state is None:
            return None
        seq_values = {} if seq_dir is None else trial_sequence_values(seq_dir.info(), trial_dir.info(), recording.device_id)
        mode = dev_state.get('mode', recording.clamp_mode)
        holding = info.get('DAQ', {}).get('command', {}).get('holding')
        return legacy_stimulus(dev_state, seq_values, description=task_dir.shortName()[:-4],
                               units=_clamp_units(mode), holding=holding)

    def load_stimulus_items(self, recording):
        """Return a list of Stimulus instances. 
        Used with LazyLoadStimulus to parse stimuli when they are needed."""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_test_pulse(self, recording):
        """Return a PatchClampTestPulse."""
        raise NotImplementedError("Don't know how to programatically determine what is a test pulse in acq4.")

    def find_nearest_test_pulse(self, recording):
        raise NotImplementedError("Don't know how to programatically determine what is a test pulse in acq4.")


def _clamp_units(mode):
    mode = (mode or '').lower()
    return {'vc': 'V', 'ic': 'A'}.get(mode)


def recorded_stimulus(protocol_info, channel=None):
    """Return the Stimulus saved in a ``.ma`` file's ``info[-1]['Protocol']``, or None.

    Clamp devices save it at ``Protocol['stimulus']``; DAQGeneric devices at
    ``Protocol[<channel>]['stimulus']``. *channel* picks the DAQGeneric channel; if None, the
    single channel that has one is used, and NotImplementedError is raised if several do.
    """
    if protocol_info.get('stimulus') is not None:
        return stimuli.Stimulus.load(protocol_info['stimulus'])
    saved = {
        ch: cmd['stimulus'] for ch, cmd in protocol_info.items()
        if isinstance(cmd, dict) and cmd.get('stimulus') is not None
    }
    if channel is not None:
        state = saved.get(channel)
    elif len(saved) > 1:
        raise NotImplementedError(f'Recording has stimuli on several channels: {sorted(saved)}')
    else:
        state = next(iter(saved.values()), None)
    return None if state is None else stimuli.Stimulus.load(state)


def trial_sequence_values(seq_info, trial_info, device):
    """Return ``{param_name: value}`` of *device*'s sequence parameters for one trial.

    *seq_info* is the ProtocolSequence dir info (``sequenceParams`` maps ``(device, name)`` to
    the list of values); *trial_info* is the trial dir info (maps ``(device, name)`` to an index).
    """
    values = {}
    for key, vals in seq_info.get('sequenceParams', {}).items():
        if isinstance(key, tuple) and key[0] == device and key in trial_info:
            values[key[1]] = vals[trial_info[key]]
    return values


def legacy_stimulus(dev_state, seq_values=None, description='stimulus', units=None, holding=None):
    """Rebuild a Stimulus from a legacy (unversioned) simple-mode stimulus generator state.

    *dev_state* is ``info['devices'][<device>]`` of the Protocol or ProtocolSequence dir; the
    generator is read from ``waveGeneratorWidget`` or ``daqState.channels.command.waveGeneratorWidget``.
    *seq_values* maps device sequence names (``'Pulse_length'``, ``'command.Pulse_length'``) to this
    trial's values; sequenced parameters missing from it take their default ``value``.
    *holding* is the recorded command holding (``.ma`` ``info[-1]['DAQ']['command']['holding']``);
    it becomes an Offset when nonzero. If None, ``holdingSpin`` is used when ``holdingCheck`` is set.
    Returns None if there is no generator state. Raises NotImplementedError for advanced-mode or
    versioned states and unknown item types.
    """
    seq_values = seq_values or {}
    chan_state, prefix = dev_state, ''
    if 'waveGeneratorWidget' not in chan_state:
        chan_state = dev_state.get('daqState', {}).get('channels', {}).get('command', {})
        prefix = 'command.'
    gen = chan_state.get('waveGeneratorWidget')
    if gen is None:
        return None
    stim_state = gen.get('stimuli')
    if gen.get('advancedMode', False) or stim_state is None:
        raise NotImplementedError('Cannot reconstruct an advanced-mode (function string) stimulus.')
    if 'format' in stim_state:
        raise NotImplementedError(f"Stimulus generator state format {stim_state['format']!r} without a saved stimulus.")

    if holding is None and chan_state.get('holdingCheck'):
        holding = chan_state['holdingSpin']
    items = []
    if holding:
        items.append(stimuli.Offset(holding, description='holding', units=units))
    for name, item in stim_state.items():
        items.append(_legacy_item(name, item, prefix, seq_values, units))
    return stimuli.Stimulus(description, items=items, units=units)


def _legacy_item(name, item, prefix, seq_values, units):
    def value(param):
        return seq_values.get(f'{prefix}{name}_{param}', item[param]['value'])

    start, length, amp = value('start'), value('length'), value('amplitude')
    # a sequenced 'sum' (length * amplitude) overrides the parameter it affects
    sum_key = f'{prefix}{name}_sum'
    if sum_key in seq_values:
        if item['sum'].get('affect', 'length') == 'length':
            length = seq_values[sum_key] / amp
        else:
            amp = seq_values[sum_key] / length

    kind = item.get('type')
    if kind == 'pulse':
        return stimuli.SquarePulse(start, length, amp, description=name, units=units)
    if kind == 'pulseTrain':
        interval = value('period') if 'period' in item else value('interpulse_length')
        n_pulses = int(round(value('pulse_number')))
        return stimuli.SquarePulseTrain(start, n_pulses, length, amp, interval, description=name, units=units)
    raise NotImplementedError(f'Unknown legacy stimulus type {kind!r} for item {name!r}.')
