from datetime import datetime
from neuroanalysis.data.loaders.loaders import DatasetLoader
from acq4.util import DataManager
from acq4.analysis.dataModels import PatchEPhys
from neuroanalysis.data.dataset import Recording, RecordingSequence, SyncRecording, TSeries, PatchClampRecording

class Acq4DatasetLoader(DatasetLoader):

    def __init__(self, filepath):

        self._filepath = filepath

        self._dh = None ## directory handle

    @property
    def dh(self):
        if self._dh is None:
            self._dh = DataManager.getDirHandle(self._filepath)
        return self._dh

    def get_sync_recordings(self, dataset):
        """Return a list of SyncRecordings."""

        sync_recs = []
        sequences = []

        for seq in self.dh.subDirs():
            if self.dh[seq].info().get('dirType') == 'ProtocolSequence':    
                sequence = RecordingSequence(parent=dataset, name=seq) 
                for sd in self.dh[seq].subDirs():
                    srec = SyncRecording(parent=dataset, key=(seq,sd))
                    sequence.add_sync_rec(srec)
                    sync_recs.append(srec)
                sequences.append(sequence)
            elif self.dh[seq].info().get('dirType') == 'Protocol':
                srec = SyncRecording(parent=dataset, key=(seq))
                sync_recs.append(srec)
            elif self.dh[seq].shortName() == 'Patch': ## ignore this for now -- how should this data be represented?
                continue
            else:
                raise Exception('Not sure how to handle folder %s' % self.dh[seq].name())

        return (sync_recs, sequences)
        

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
                meta = {}
                meta['file_name'] = f.name()
                meta['clamp_mode'] = PatchEPhys.getClampMode(f).lower()
                if meta['clamp_mode'] == 'vc':
                    meta['holding_current'] = PatchEPhys.getClampHoldingLevel(f)
                elif meta['clamp_mode'] == 'ic':
                    meta['holding_potential'] = PatchEPhys.getClampHoldingLevel(f)
                    meta['bridge_balance'] = PatchEPhys.getBridgeBalanceCompensation(f)
                else:
                    raise Exception("dont know how to interpret %s clamp_mode" % meta['clamp_mode'])

                data = f.read()
                dt = data.axisValues(1)[1] - data.axisValues(1)[0]

                rec = PatchClampRecording(
                    channels={'primary':TSeries(channel_id='primary', data=data['primary'].asarray(), dt=dt, units=data.columnUnits(0, 'primary')),
                              'command':TSeries(channel_id='command', data=data['command'].asarray(), dt=dt, units=data.columnUnits(0, 'command'))},
                    start_time=start_time, 
                    device_type='patch clamp amplifier', 
                    device_id=None,
                    sync_recording=sync_rec,
                    **meta
                    )
                rec['primary']._recording = rec
                rec['command']._recording = rec

                recordings[f.name(relativeTo=dh)] = rec

            else:
                data = f.read()
                time_axis = data._getAxis('Time') ## is there a way to do this without using a private MA function?
                if 'Channel' in data.listColumns().keys():
                    channel_axis = data._getAxis('Channel')

                #dt = data.axisValues(time_axis)[1] - data.axisValues(time_axis)[0]

                if data.ndim == 2:
                    channels = {k:TSeries(channel_id=k, data=data[k].asarray(), time_values=data.axisValues(time_axis), units=data.columnUnits(channel_axis, k)) for k in data.listColumns(channel_axis)}
                elif data.ndim == 3:
                    channels = {'frames':TSeries(channel_id='frames', data=data.asarray(), time_values=data.axisValues(time_axis))}

                rec = Recording(
                    channels=channels,
                    start_time=start_time,
                    device_type=f.name(relativeTo=dh).strip('.ma'),
                    sync_recording=sync_rec,
                    file_name=f.name(),
                    **data.infoCopy()[-1]
                    )

                for k in rec.channels:
                    rec[k]._recording = rec

                recordings[f.name(relativeTo=dh)] = rec

        return recordings


    def get_tseries_data(self, tseries):
        """Return a numpy array of the data in the tseries."""
        #### I don't think we need this because we hand TSeries their data when we instantiate them.
        raise NotImplementedError("Must be implemented in subclass.")

    def load_stimulus(self, recording):
        """Return an instance of stimuli.Stimulus"""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_stimulus_items(self, recording):
        """Return a list of Stimulus instances. 
        Used with LazyLoadStimulus to parse stimuli when they are needed."""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_test_pulse(self, recording):
        """Return a PatchClampTestPulse."""
        raise NotImplementedError("Must be implemented in subclass.")

    def find_nearest_test_pulse(self, recording):
        raise NotImplementedError("Must be implemented in subclass.")