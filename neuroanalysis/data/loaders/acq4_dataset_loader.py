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

    def load_stimulus(self, recording) -> stimuli.Stimulus:
        #### I don't know whether I should try to parse this from metadata, or just find square pulses in the command waveform.
        ### I think finding square pulses would be simpler, but makes the assumption that pulses are square. Which is probably usually true.
        ### what if I check the wavegenerator widget data for the function name (pulse) and then findSquarepulses, or raise an exception if it's a different function?
        if not isinstance(recording, PatchClampRecording):
            raise NotImplementedError('not implemented yet')
        fh = DataManager.getFileHandle(recording.meta['file_name'])
        seqDir = PatchEPhys.getParent(fh, 'ProtocolSequence')
        if seqDir is not None:
            dev_info = seqDir.info()['devices'][recording.device_id]

            if dev_info['mode'].lower() == 'vc':
                units = 'V'
            elif dev_info['mode'].lower() == 'ic':
                units = 'A'
            else:
                units = None

            items = []

            if dev_info['holdingCheck']:
                items.append(stimuli.Offset(dev_info['holdingSpin']))

            stim_pulses = PatchEPhys.getStimParams(fh)

            for p in stim_pulses:
                if p['function_type'] == 'pulse':
                    items.append(stimuli.SquarePulse(p['start'], p['length'], p['amplitude']))
                elif p['function_type'] == 'pulseTrain':
                    items.append(stimuli.SquarePulseTrain(p['start'], p['pulse_number'], p['length'], p['amplitude'], p['period']))

            desc = seqDir.shortName()[:-4]
            return stimuli.Stimulus(desc, items=items, units=units)

    def load_stimulus_items(self, recording):
        """Return a list of Stimulus instances. 
        Used with LazyLoadStimulus to parse stimuli when they are needed."""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_test_pulse(self, recording):
        """Return a PatchClampTestPulse."""
        raise NotImplementedError("Don't know how to programatically determine what is a test pulse in acq4.")

    def find_nearest_test_pulse(self, recording):
        raise NotImplementedError("Don't know how to programatically determine what is a test pulse in acq4.")