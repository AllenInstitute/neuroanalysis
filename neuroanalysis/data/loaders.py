import h5py
import numpy as np
from collections import OrderedDict
from neuroanalysis.data.dataset import SyncRecording, PatchClampRecording, Recording, TSeries
import neuroanalysis.util.mies_nwb_parsing as parser
#import aisynphys.pipeline.opto.data_model as dm
import neuroanalysis.util.device_config as dm


class MiesNwbLoader():

    def __init__(self, file_path):
        self._file_path = file_path

        self._time_series = None ## parse nwb into sweep_number: info dictionary for lookup of individual sweeps
        self._notebook = None ## parse the lab_notebook part of the nwb 
        self._hdf = None ## holder for the .hdf file
        self._rig = None ## holder for the name of the rig this nwb was recorded on

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

    @property
    def rig(self):
        if self._rig is None:
            self._rig = dm.get_rig_from_nwb(notebook=self.notebook)
        return self._rig

    def get_sync_recordings(self, dataset):
        ### miesnwb parses sweeps and contents into nwb._timeseries -- this happens in a hidden way inside nwb.contents()
        ## other classes (sync_recordings, etc) then use nwb._timeseries to look up their data by sweep number
        ## So, possibly we save _timeseries here in the loader instead.
        sweep_ids = sorted(list(self.time_series.keys()))
        sweeps = []
        for sweep_id in sweep_ids:
            sweeps.append(SyncRecording(parent=dataset, key=sweep_id))
        return sweeps 


    def get_recordings(self, sync_rec):
        ### return {device: recording}
        recordings = {}
        sweep_id = sync_rec.key

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
                        channels={'primary':TSeries(channel_id='primary', dt=dt, start_time=start_time), 
                                  'command':TSeries(channel_id='command', dt=dt, start_time=start_time)},
                        start_time=start_time,
                        device_type="MultiClamp 700",
                        device_id=device_id,
                        sync_recording=sync_rec,
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
                    device = 'Fidelity' ## do this for right now, implement lookup in the future

                    rec = Recording(
                        #channels = {'reporter':TSeries(data=np.array(data), dt=dt)},
                        channels = {'reporter':TSeries(channel_id='reporter', dt=dt, start_time=start_time)},
                        device_type = device, 
                        device_id=device, 
                        sync_recording = sync_rec,
                        **meta)
                    rec['reporter']._recording = rec

                    recordings[rec.device_id] = rec

            ## now get associated ttl traces:
            for k in self.hdf['stimulus/presentation'].keys():
                if k.startswith('data_%05d_TTL' % sweep_id):
                    ttl_data = self.hdf['stimulus/presentation/' + k]['data']
                    dt = ttl_data.attrs['IGORWaveScaling'][1,0] / 1000.

                    ttl_num = k.split('_')[-1]
                    device = dm.device_mapping[self.rig]['TTL1_%s'%ttl_num]

                    meta={}
                    meta['sweep_name'] = k

                    rec = Recording(
                        channels={'reporter':TSeries(channel_id='reporter', dt=dt)},
                        device_type = device,
                        device_id=device,
                        sync_recording=sync_rec,
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


        if np.isnan(data[-1]):
            # recording was interrupted; remove NaNs from the end of the array
            last_sample = np.argwhere(np.isfinite(data)).max()
            data = data[:last_sample+1]

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







