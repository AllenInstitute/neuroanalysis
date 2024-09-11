import h5py
import numpy as np

# import aisynphys.pipeline.opto.data_model as dm
import neuroanalysis.stimuli as stimuli
import neuroanalysis.util.mies_nwb_parsing as parser
from neuroanalysis.data.dataset import SyncRecording, PatchClampRecording, Recording, TSeries
from neuroanalysis.data.loaders.loaders import DatasetLoader
from neuroanalysis.test_pulse import PatchClampTestPulse


class MiesNwbLoader(DatasetLoader):
    _baseline_analyzer_class = None ## make room for subclasses to automatically supply baseline analyzers

    def __init__(self, file_path, baseline_analyzer_class=None):
        self._file_path = file_path
        if baseline_analyzer_class is not None:
            self._baseline_analyzer_class = baseline_analyzer_class

        self._time_series = None ## parse nwb into sweep_number: info dictionary for lookup of individual sweeps
        self._notebook = None ## parse the lab_notebook part of the nwb 
        self._hdf = None ## holder for the .hdf file
        self._rig = None ## holder for the name of the rig this nwb was recorded on
        self._device_config = None 

    @property
    def hdf(self):
        if self._hdf is None:
            self._hdf = h5py.File(self._file_path, 'r')
        return self._hdf

    @property
    def time_series(self):
        if self._time_series is None:
            self._time_series = {}
            for ts_name, ts in self.hdf['acquisition/timeseries'].items():
                src = dict([field.split('=') for field in ts.attrs['source'].split(';')])
                sweep = int(src['Sweep'])
                ad_chan = int(src['AD'])
                src['hdf_group_name'] = 'acquisition/timeseries/' + ts_name
                self._time_series.setdefault(sweep, {})[ad_chan] = src
        return self._time_series

    @property
    def notebook(self):
        """Return compiled data from the lab notebook.

        The format is a dict like ``{sweep_number: [ch1, ch2, ...]}`` that contains one key:value
        pair per sweep. Each value is a list containing one metadata dict for each channel in the
        sweep. For example::

            nwb.notebook()[sweep_id][channel_id][metadata_key]
        """
        if self._notebook is None:
            self._notebook = parser.parse_lab_notebook(self.hdf)
        return self._notebook

    def get_dataset_name(self):
        return self._file_path

    def get_sync_recordings(self, dataset):
        ### miesnwb parses sweeps and contents into nwb._timeseries -- this happens in a hidden way inside nwb.contents()
        ## other classes (sync_recordings, etc) then use nwb._timeseries to look up their data by sweep number
        ## So, possibly we save _timeseries here in the loader instead.
        sweep_ids = sorted(list(self.time_series.keys()))
        sweeps = []
        for sweep_id in sweep_ids:
            sweeps.append(SyncRecording(parent=dataset, key=sweep_id, loader=self))
        return sweeps 

    def get_recordings(self, sync_rec):
        ### return {device: recording}
        recordings = {}
        sweep_id = sync_rec.key

        ### Hardcode this now, figure out configuration system when needed
        device_map = {
            'AD6': 'Fidelity',
            'TTL1_0': 'Prairie_Command',
            'TTL1_1': 'LED-470nm',
            'TTL1_2': 'LED-590nm'
        }

        for ch, meta in self.time_series[sweep_id].items():
            if 'data_%05d_AD%d' %(sweep_id, ch) in self.hdf['acquisition/timeseries'].keys():
                hdf_group = self.hdf['acquisition/timeseries/data_%05d_AD%d' %(sweep_id, ch)]

                ### this channel is a patch-clamp headstage
                if 'electrode_name' in hdf_group: 
                    #rec = OptoMiesRecording(self, sweep_id, ch)
                    device_id = int(hdf_group['electrode_name'][()][0].split('_')[1])

                    nb = self.notebook[sweep_id][device_id]
                    meta = {}
                    meta['holding_potential'] = (
                        None if nb['V-Clamp Holding Level'] is None
                        else nb['V-Clamp Holding Level'] * 1e-3
                    )
                    meta['holding_current'] = (
                        None if nb['I-Clamp Holding Level'] is None
                        else nb['I-Clamp Holding Level'] * 1e-12
                    )   
                    meta['notebook'] = nb
                    if nb['Clamp Mode'] == 0:
                        meta['clamp_mode'] = 'vc'
                    else:
                        meta['clamp_mode'] = 'ic'
                        meta['bridge_balance'] = (
                            0.0 if nb['Bridge Bal Enable'] == 0.0 or nb['Bridge Bal Value'] is None
                            else nb['Bridge Bal Value'] * 1e6
                        )
                    meta['lpf_cutoff'] = nb['LPF Cutoff']
                    offset = nb['Pipette Offset']  # sometimes the pipette offset recording can fail??
                    meta['pipette_offset'] = None if offset is None else offset * 1e-3
                    meta['sweep_name'] = 'data_%05d_AD%d' %(sweep_id, ch)
                    start_time = parser.igorpro_date(nb['TimeStamp'])
                    dt = hdf_group['data'].attrs['IGORWaveScaling'][1,0] / 1000.


                    rec = PatchClampRecording(### this makes TSeries when we make Recordings instead of waiting until recordings ask for their TSeries -- which is something I've been trying to get away from in the rest of this refactor
                        channels={'primary':TSeries(channel_id='primary', dt=dt, start_time=start_time, loader=self), 
                                  'command':TSeries(channel_id='command', dt=dt, start_time=start_time, loader=self)},
                        start_time=start_time,
                        device_type="MultiClamp 700",
                        device_id=device_id,
                        sync_recording=sync_rec,
                        loader=self,
                        **meta
                        )
                    rec['primary']._recording = rec
                    rec['command']._recording = rec

                    recordings[rec.device_id] = rec


                ### Alice checked to see if there were pulses before labeling a trace as fidelity or ttl - if there weren't pulses she labelled it unknown -- is this necessary? -- I'm gonna say 'no' for right now
                else: ### This is a pockel-cell recording
                    dt = hdf_group['data'].attrs['IGORWaveScaling'][1,0]/1000.
                    nb = self.notebook[sweep_id][ch] ## not sure if ch is the right thing to access this
                    meta = {}
                    meta['notebook'] = nb
                    meta['sweep_name'] = 'data_%05d_AD%d'%(sweep_id, ch)
                    start_time = parser.igorpro_date(nb['TimeStamp'])
                    #device = 'Fidelity' ## do this for right now, implement lookup in the future
                    device = device_map[meta['sweep_name'][-3:]]

                    rec = Recording(
                        #channels = {'reporter':TSeries(data=np.array(data), dt=dt)},
                        channels = {'reporter':TSeries(channel_id='reporter', dt=dt, start_time=start_time, loader=self)},
                        device_type = device, 
                        device_id=device, 
                        sync_recording = sync_rec,
                        loader=self,
                        start_time=start_time,
                        **meta)
                    rec['reporter']._recording = rec

                    recordings[rec.device_id] = rec

            ## now get associated ttl traces:
            for k in self.hdf['stimulus/presentation'].keys():
                if k.startswith('data_%05d_TTL' % sweep_id):
                    ttl_data = self.hdf['stimulus/presentation/' + k]['data']
                    dt = ttl_data.attrs['IGORWaveScaling'][1,0] / 1000.

                    #ttl_num = k.split('_')[-1]
                    #device = self.device_config['TTL1_%s'%ttl_num]
                    ttl = k.split('_', 2)[-1]
                    device = device_map[ttl]

                    meta={}
                    meta['sweep_name'] = k

                    rec = Recording(
                        channels={'reporter':TSeries(channel_id='reporter', dt=dt, loader=self)},
                        device_type = device,
                        device_id=device,
                        sync_recording=sync_rec,
                        loader=self,
                        **meta)
                    rec['reporter']._recording = rec

                    recordings[rec.device_id]=rec


                            #     rec.device_name = device_mapping['Wayne']['AD%d'%ch]
                #     return rec
                # else:
                #     k = 'data_%05d_AD%d' % (sweep_id, ch)
                #     opto_rec = OptoRecording(self, sweep_id, ch, k)
                #     if opto_rec is None:
                #         return None
                #     else:
                #         opto_rec.device_name = device_mapping['Wayne']['AD%d'%ch]
                #         return opto_rec

        return recordings


    def get_tseries_data(self, tseries):
        rec = tseries.recording
        chan = tseries.channel_id

        if chan == 'primary':
            scale = 1e-12 if rec.clamp_mode == 'vc' else 1e-3
            #data = np.array(rec.primary_hdf) * scale
            data = np.array(self.hdf['acquisition']['timeseries'][rec.meta['sweep_name']]['data'])*scale

        elif chan == 'command':
            scale = 1e-3 if rec.clamp_mode == 'vc' else 1e-12
            # command values are stored _without_ holding, so we add
            # that back in here.
            offset = rec.holding_potential if rec.clamp_mode == 'vc' else rec.holding_current
            if offset is None:
                exc = Exception("Holding value unknown for this recording; cannot generate command data.")
                # Mark this exception so it can be ignored in specific places
                exc._ignorable_bug_flag = True
                raise exc
            #self._data = (np.array(rec.command_hdf) * scale) + offset
            data = (np.array(self.hdf['stimulus']['presentation']['data_%05d_DA%d'%(rec.sync_recording.key, self.get_da_chan(rec))]['data']) * scale) + offset

        elif chan == 'reporter':
            if 'AD' in rec.meta['sweep_name']:
                data = np.array(self.hdf['acquisition']['timeseries'][rec.meta['sweep_name']]['data'])
            elif 'TTL' in rec.meta['sweep_name']:
                data = np.array(self.hdf['stimulus']['presentation'][rec.meta['sweep_name']]['data'])
            else:
                raise Exception("Not sure where to find data for recording: %s"%rec.meta['sweep_name'])

        else:
            raise Exception("Getting data for channels named %s is not yet implemented." % chan)


        if np.isnan(data[-1]):
            # recording was interrupted; remove NaNs from the end of the array

            first_nan = np.searchsorted(data, np.nan)
            data = data[:first_nan]

        return data

    def get_da_chan(self, rec):
        """Return the DA channel ID for the given recording.
        """
        da_chan = None

        hdf = self.hdf['stimulus/presentation']
        stims = [k for k in hdf.keys() if k.startswith('data_%05d_'%rec.sync_recording.key)]
        for s in stims:
            if 'TTL' in s:
                continue
            elec = hdf[s]['electrode_name'][()][0]
            if elec == 'electrode_%d' % rec.device_id:
                da_chan = int(s.split('_')[-1][2:])

        if da_chan is None:
            raise Exception("Cannot find DA channel for headstage %d" % self.device_id)

        return da_chan

    def load_test_pulse(self, rec):
        if not isinstance(rec, PatchClampRecording):
            raise TypeError(f"Can only load test pulses for PatchClampRecording, not {type(rec)}")

        if rec.meta['notebook']['TP Insert Checkbox'] != 1.0:  # no test pulse
            return None

        # get start/stop indices of the test pulse region
        pulse_dur = rec.meta['notebook']['TP Pulse Duration'] / 1000.
        total_dur = pulse_dur / (1.0 - 2. * rec.meta['notebook']['TP Baseline Fraction'])
        start = 0
        stop = start + int(total_dur / rec['primary'].dt)

        return PatchClampTestPulse(rec, indices=(start, stop))

    def find_nearest_test_pulse(self, rec):
        sweep_id = rec.sync_recording.key
        device_id = rec.device_id

        min_dt = None
        nearest = None
        for srec in rec.sync_recording.parent.contents:
            if device_id not in srec.devices:
                continue
            if srec[device_id].meta['notebook']['TP Insert Checkbox'] == 1.0:
                dt = abs((srec[device_id].start_time - rec.start_time).total_seconds())
                if min_dt is None or dt < min_dt:
                    min_dt = dt
                    nearest = srec[device_id].test_pulse
                if min_dt is not None and dt > min_dt:
                    break

        return nearest

    def load_stimulus(self, rec):
        if isinstance(rec, PatchClampRecording):
            desc = self.hdf['acquisition/timeseries'][rec.meta['sweep_name']]['stimulus_description'][()][0]
            return stimuli.LazyLoadStimulus(description=desc, loader=self, source=rec)
        else:
            raise Exception('not implemented yet')
        #return stimuli.Stimulus(description=desc, items=self.load_stimulus_items(rec))

    def load_stimulus_items(self, rec):
        items = []

        # Add holding offset, determine units
        if rec.clamp_mode == 'ic':                
            units = 'A'
            items.append(stimuli.Offset(
                start_time=0,
                amplitude=rec.holding_current,
                description="holding current",
                units=units,
            ))
        elif rec.clamp_mode == 'vc':
            units = 'V'
            items.append(stimuli.Offset(
                start_time=0,
                amplitude=rec.holding_potential,
                description="holding potential",
                units=units,
            ))
        else:
            units = None
        
        # inserted test pulse?
        #if rec.has_inserted_test_pulse:
        #    self.append_item(rec.inserted_test_pulse.stimulus)
        if rec.test_pulse is not None:
            items.append(rec.test_pulse.stimulus)

        notebook = rec.meta['notebook']
        
        if 'Stim Wave Note' in notebook:
            # Stim Wave Note format is explained here: 
            # https://alleninstitute.github.io/MIES/file/_m_i_e_s___wave_builder_8ipf.html#_CPPv319WB_GetWaveNoteEntry4wave8variable6string8variable8variable

            # read stimulus structure from notebook
            #version, epochs = rec._stim_wave_note()
            version, epochs = parser.parse_stim_wave_note(notebook)
            assert len(epochs) > 0
            scale = (1e-3 if rec.clamp_mode == 'vc' else 1e-12) * notebook['Stim Scale Factor']
            t = (notebook['Delay onset oodDAQ'] + notebook['Delay onset user'] + notebook['Delay onset auto']) * 1e-3
            
            # if dDAQ is active, add delay from previous channels
            if notebook['Distributed DAQ'] == 1.0:
                ddaq_delay = notebook['Delay distributed DAQ'] * 1e-3
                for dev in rec.parent.devices:
                    other_rec = rec.parent[dev]
                    if other_rec is rec:
                        break
                    #_, epochs = rec._stim_wave_note()
                    if 'Stim Wave Note' in other_rec.meta['notebook']:
                        _, other_epochs = parser.parse_stim_wave_note(other_rec.meta['notebook'])
                        for ep in other_epochs:
                            dt = float(ep.get('Duration', 0)) * 1e-3
                            t += dt
                        t += ddaq_delay
            
            for epoch_n,epoch in enumerate(epochs):
                try:
                    if epoch['Epoch'] == 'nan':
                        # Sweep-specific entry; not sure if we need to do anything with this.
                        continue

                    stim_type = epoch.get('Type')
                    duration = float(epoch.get('Duration', 0)) * 1e-3
                    name = "Epoch %d" % int(epoch['Epoch'])
                    if stim_type == 'Square pulse':
                        item = stimuli.SquarePulse(
                            start_time=t, 
                            amplitude=float(epoch['Amplitude']) * scale, 
                            duration=duration, 
                            description=name,
                            units=units,
                        )
                    elif stim_type == 'Pulse Train':
                        assert epoch['Poisson distribution'] == 'False', "Poisson distributed pulse train not supported"
                        assert epoch['Mixed frequency'] == 'False', "Mixed frequency pulse train not supported"
                        assert epoch['Pulse Type'] == 'Square', "Pulse train with %s pulse type not supported"
                        item = stimuli.SquarePulseTrain(
                            start_time=t,
                            n_pulses=int(epoch['Number of pulses']),
                            pulse_duration=float(epoch['Pulse duration']) * 1e-3,
                            amplitude=float(epoch['Amplitude']) * scale,
                            interval=float(epoch['Pulse To Pulse Length']) * 1e-3,
                            description=name,
                            units=units,
                        )
                    elif stim_type == 'Sin Wave':
                        # bug in stim wave note version 2: log chirp field is inverted
                        is_chirp = epoch['Log chirp'] == ('False' if version <= 2 else 'True')
                        if is_chirp:
                            assert epoch['FunctionType'] == 'Sin', "Chirp wave function type %s not supported" % epoch['Function type']
                            item = stimuli.Chirp(
                                start_time=t,
                                start_frequency=float(epoch['Frequency']),
                                end_frequency=float(epoch['End frequency']),
                                duration=duration,
                                amplitude=float(epoch['Amplitude']) * scale,
                                phase=0,
                                offset=float(epoch['Offset']) * scale,
                                description=name,
                                units=units,
                            )
                        else:
                            if epoch['FunctionType'] == 'Sin':
                                phase = 0
                            elif epoch['FunctionType'] == 'Cos':
                                phase = np.pi / 2.0
                            else:
                                raise ValueError("Unsupported sine wave function type: %r" % epoch['FunctionType'])
                                
                            item = stimuli.Sine(
                                start_time=t,
                                frequency=float(epoch['Frequency']),
                                duration=duration,
                                amplitude=float(epoch['Amplitude']) * scale,
                                phase=phase,
                                offset=float(epoch['Offset']) * scale,
                                description=name,
                                units=units,
                            )
                    else:
                        print(epoch)
                        print("Warning: unknown stimulus type %s in %s sweep %s" % (stim_type, self._file_path, rec.meta['sweep_name']))
                        item = None
                except Exception as exc:
                    print("Warning: error reading stimulus epoch %d in %s sweep %s: %s" % (epoch_n, self._file_path, rec.meta['sweep_name'], str(exc)))
            
                t += duration
                if item is not None:
                    items.append(item)

        return items

    def get_baseline_regions(self, recording):
        if self._baseline_analyzer_class is None:
            raise Exception("Cannot get baseline regions, no baseline analyzer class was supplied upon initialization of %s." % self.__class__.__name__)

        return self._baseline_analyzer_class.get(recording.sync_recording).baseline_regions





### Notes:
# -- Goal: make it so we don't have subclasses of the data model classes - ie, use Dataset, SyncRecording, Recording, PatchClampRecording -- instead move idiosyncracies about loading data into a DataLoader class
# -- one problem I'm running into is with all the extra functions implemented in MiesRecording - like nearest_test_pulse and baseline_regions 
#        -- on the one hand I wonder if that is the right place for those functions, and on the other I'm not sure where to put them
#              -> well, I think maybe baseline regions should be an analyzer. Maybe nearest test pulse should be as well.
# -- part of why I'm running into the problems with the extra functions is that I don't know how to handle all the info in the lab_notebook, and those
#      functions use the lab_notebook info
# -- solution (?):
#       1) Parse the notebook info and save it in the loader.
#       2) Use the notebook info to supply standard metadata about each recording and save the rest in rec.meta (anything should be allowed in rec.meta)
#       3) Write an AI specific analyzer that returns the nearest_test_pulse for a recording (and because this is AI specific it can raise errors if
#           the correct metadata isn't there)
#       4) Write a baseline_regions analyzer, and use that to get baseline_regions instead of that analysis happening inside the Recording class.







