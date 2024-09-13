"""
Data Abstraction Layer
----------------------

These classes present a high-level API for handling data from various types of neurophysiology experiments.
They do not implement functionality for reading any particular type of data file, though. Rather, they
provide an abstraction layer that separates analysis and visualization from the idiosyncratic
details of any particular data acquisition system.

Each data acquisition system can provide a set of subclasses to adapt its data formats to this API
(as long as the data are reasonably similar), and analysis tools should rely only on this API to ensure
that they will not need to be rewritten in order to support data collected under different acquisition
systems.

This abstraction layer also helps to enforce good coding practice by separating data representation,
analysis, and visualization.
"""
from collections import OrderedDict

import numpy as np

from ..baseline import float_mode
from ..filter import downsample


class Container(object):
    """Generic hierarchical container. 
    
    This class is the basis for most other classes in the DAL.
    """
    def __init__(self, loader=None):
        self._meta = OrderedDict()
        self._key = None
        self._parent = None
        self._loader = loader

    @property
    def loader(self):
        if self._loader is None:
            raise Exception("No loader was specified upon initialization.")
        return self._loader
        
    @property
    def parent(self):
        return self._parent

    # @property
    # def top_parent(self):
    #     if self.parent is not None:
    #         return self.parent.top_parent
    #     return self
    
    @property
    def children(self):
        return []

    @property
    def key(self):
        """Key that uniquely identifies this object among its siblings.
        """
        return self._key
    
    @property
    def meta(self):
        return self._meta

    def update_meta(self, **kargs):
        self._meta.update(kargs)

    @property
    def all_children(self):
        allch = [self]
        for ch in self.children:
            allch.extend(ch.all_children)
        return allch

    @property
    def all_meta(self):
        allmeta = OrderedDict()
        for obj in self.path:
            m = obj.meta
            allmeta.update(m)
        return allmeta
        
    @property
    def path(self):
        obj = self
        path = []
        while obj is not None:
            path.append(obj)
            obj = obj.parent
        return path[::-1]


class Dataset(Container):
    """A generic container for RecordingSequence, SyncRecording, Recording, and TSeries instances that
    were acquired together.
    
    The boundaries between one experiment and the next are sometimes ambiguous, but
    in general we group multiple recordings into an experiment if they are likely to
    be analyzed together. Likewise, recordings that have no causal relationship
    to each other probably belong in different Dataset containers. For example,
    a series of recordings made on the same cell almost certainly belong in the same
    Dataset, whereas recordings made from different pieces of tissue probably
    belong in different Datasets.

    Parameters
    ----------
    data : list of SyncRecordings | None
        A list of SyncRecordings in the dataset. If None, they will be loaded when needed with loader
    meta : dict | None
        An optional dict of metadata about the dataset.
    loader : a class | None
        A class which contains functions for loading data. Must include:
            get_sync_recordings(self) - Return a tuple of (list of SyncRecordings, list of RecordingSequences) in this dataset
            get_dataset_name(self) - Return the name of this dataset (str)
    sequences: list of RecordingSequences | None
        A list of RecordingSequences in this dataset. If None they will be loaded when needed with loader
    name : str | None
        The name of this dataset. If None, will be loaded when needed with loader

    """

    def __init__(self, data=None, meta=None, loader=None, name=None):
        Container.__init__(self, loader=loader)
        self._data = data
        self._name = name
        if meta is not None:
            self._meta.update(OrderedDict(meta))

        #self._loader = loader
        #self._sequences = sequences
    
    @property
    def contents(self):
        """A list of SyncRecording objects directly contained in this experiment.
        
        Grandchild objects are not included in this list.
        """
        if self._data is None:
            self._data = self.loader.get_sync_recordings(self)
        return self._data[:]

    # @property
    # def sequences(self):
    #     if self._sequences is None:
    #         self.contents
    #     return self._sequences

    @property
    def name(self):
        if self._name is None:
            self._name = self.loader.get_dataset_name()
        return self._name

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.name)

    def find(self, type):
        return [c for c in self.all_children if isinstance(c, type)]

    @property
    def all_traces(self):
        return self.find(TSeries)
    
    @property
    def all_recordings(self):
        return self.find(Recording)

    @property
    def all_sync_recordings(self):
        return self.find(SyncRecording)

    def meta_table(self, objs):
        # collect all metadata
        meta = []
        for i,o in enumerate(objs):
            meta.append(o.all_meta)
            
        # generate a list of common fields (in the correct order)
        fields = set(meta[0].keys())
        for m in meta[1:]:
            fields &= set(m.keys())
        order = list(meta[0].keys())
        for k in order[:]:
            if k not in fields:
                order.remove(k)
        
        # transpose
        tr = OrderedDict()
        for k in order:
            tr[k] = [m[k] for m in meta]
        
        # create a table
        import pandas
        return pandas.DataFrame(tr)

    @property
    def trace_table(self):
        return self.meta_table(self.all_traces)

    # @property
    # def parent(self):
    #     """None
        
    #     This is a convenience property used for traversing the object hierarchy.
    #     """
    #     return None
    
    @property
    def children(self):
        """Alias for self.contents
        
        This is a convenience property used for traversing the object hierarchy.
        """
        return self.contents
    
    
# class RecordingSequence(Container):

# For now, possibly forever, remove the concept of Sequences from datasets. 
# It is too hard to decide programatically which sync recordings should make up a sequence,
# and it is likely that that will change or be highly specific to each type of experiment,
# and will likely involve manual annotation.
# So, I think that sorting sync recordings into sequences should be something that happens at a 
# higher level than loading datasets. 
    
#     #  Acquisition?
#     #  RecordingSet?
    
#     #  Do we need both SyncRecordingSequence and RecordingSequence ?
#     #  Can multiple RecordingSequence instances refer to the same underlying sequence?
#     #    - Let's say no--otherwise we have to worry about unique identification, comparison, etc.
#     #    - Add ___View classes that slice/dice any way we like.
    
#     """Representation of a sequence of data acquisitions.

#     For example, this could be a single type of acquisition that was repeated ten times,
#     or a series of ten acquisitions that varies one parameter across ten values.
#     Usually the recordings in a sequence all use the same set of devices.

#     Sequences may be multi-dimensional and may vary more than one parameter.
    
#     Items in a sequence are usually SyncRecording instances, but may also be
#     nested RecordingSequence instances.
#     """

#     def __init__(self, parent, name, sync_recs=None, meta=None, loader=None):
#         Container.__init__(self, loader=loader)

#         self._parent = parent
#         self._key = name
#         if meta is not None:
#             self.update_meta(**meta)

#         self._sync_recs = []
#         if sync_recs is not None:
#             for sync_rec in sync_recs:
#                 self.add_sync_rec(sync_rec)

#     def __repr__(self):
#         return "<%s %s>" % (self.__class__.__name__, self.key)

#     def add_sync_rec(self, sync_rec):
#         if sync_rec not in self._sync_recs:
#             self._sync_recs.append(sync_rec)

#     @property
#     def type(self):
#         """An arbitrary string representing the type of acquisition.
#         """
#         return self._meta.get('type', None)

#     # @property
#     # def shape(self):
#     #     """The array-shape of the sequence.
#     #     """

#     # @property
#     # def ndim(self):
#     #     return len(self.shape)

#     # def __getitem__(self, item):
#     #     """Return one item (a SyncRecording instance) from the sequence.
#     #     """

#     def sequence_params(self):
#         """
#         Return a dictionary of {param_name:values}.

#         ## maybe in the future:
#         //Return a structure that describes the parameters that are varied across each
#         //axis of the sequence.

#         //For example, a two-dimensional sequence might return the following:
#         //
#         //    [
#         //        [param1, param2],   # two parameters that vary across the first sequence axis
#         //        [],  # no parameters vary across the second axis (just repetitions)
#         //        [param3],  # one parameter that varies across all recordings, regardless of its position along any axis
#         //    ]

#         //Each parameter must be a key under 'sequence_params' in the meta data for a single syncrecording.
#         """
#         return self._meta.get('sequence_params')

#     # @property
#     # def parent(self):
#     #     """None
        
#     #     This is a convenience property used for traversing the object hierarchy.
#     #     """
#     #     return None
#     @property
#     def contents(self):
#         return self._sync_recs
    
#     @property
#     def children(self):
#         """Alias for self.contents
        
#         This is a convenience property used for traversing the object hierarchy.
#         """
#         return self.contents


class SyncRecording(Container):
    """Representation of multiple synchronized recordings.

    This is typically the result of recording from multiple devices at the same time
    (for example, two patch-clamp amplifiers and a camera).

    Parameters
    ----------
    recordings : list of Recordings | None
        A list of Recordings in this sync_recording. If None, will be loaded when needed with loader.
    parent : Dataset | None
        The dataset this sync recording is part of.
    key : 
        A id for this sync recording that is unique among its siblings
    loader : a class | None
        A class which contains functions for loading data in the sync recording. Must have these methods:
            get_recordings(self) - Return a dict of {device: Recording}

    """
    def __init__(self, recordings=None, parent=None, key=None, meta=None, loader=None):
        Container.__init__(self, loader=loader)
        self._parent = parent
        self._recording_dict = recordings #if recordings is not None else OrderedDict()
        self._key = key
        if meta is not None:
            self.update_meta(**meta)

    def __repr__(self):
        return "<%s key=%s>" % (self.__class__.__name__, str(self.key))

    @property
    def type(self):
        """An arbitrary string representing the type of acquisition.
        """
        pass

    @property
    def recording_dict(self):
        if self._recording_dict is None:
            self._recording_dict = self.loader.get_recordings(self)
        return self._recording_dict

    @property
    def devices(self):
        """A list of the names of devices in this recording.
        """
        return list(self.recording_dict.keys())

    def __getitem__(self, item):
        """Return a recording given its device name.
        """
        return self.recording_dict[item]

    @property
    def recordings(self):
        """A list of the recordings in this syncrecording.
        """
        return list(self.recording_dict.values())

    def data(self):
        return np.concatenate([self[dev].data()[None, :] for dev in self.devices], axis=0)

    # @property
    # def parent(self):
    #     return self._parent

    @property
    def children(self):
        return self.recordings


device_tree = {
    'patch clamp amplifier': {
        'MultiClamp 700': [
            'MultiClamp 700A',
            'MultiClamp 700B',
        ],
    },
}


class Recording(Container):
    """Representation of a single continuous data acquisition from a single device,
    possibly with multiple channels of data (for example, a recording from a single
    patch-clamp headstage with input and output channels, or ).
    
    Each channel is described by a single TSeries instance. Channels are often 
    recorded with the same timebase, but this is not strictly required.

    Parameters
    ----------
    channels : dict | None
        A {name : TSeries} dict of the channels in this recording. 
    start_time : ? | None
        The start time of this recording.
    device_type : str | None
        The type of device that made this recording.
    device_id : ? | None
        The id of the device that made this recording.
    sync_recording : SyncRecording | None
        The parent sync recording this recording is part of.
    loader : a class | None
        A loader class for loading information not supplied at initialization. No required functions yet.

    """
    def __init__(self, channels=None, start_time=None, device_type=None, device_id=None, sync_recording=None, loader=None, **meta):
        Container.__init__(self, loader=loader)
        self._meta = OrderedDict([
            ('start_time', start_time),
            ('device_type', device_type),
            ('device_id', device_id),
        ])
        self._meta.update(meta)
        
        if channels is None:
            channels = OrderedDict()
        else:
            channels = OrderedDict(channels)
            for k,v in channels.items():
                assert isinstance(v, TSeries)
        self._channels = channels

        self._sync_recording = sync_recording
        self._parent = sync_recording
        
    @property
    def device_type(self):
        """A string representing the type of device that generated this recording.

        Strings should be described in the global ``device_tree``.        
        """
        return self._meta['device_type']

    @property
    def channels(self):
        """A list of channels included in this recording.
        """
        return list(self._channels.keys())

    @property
    def start_time(self):
        """The starting time (unix epoch) of this recording.
        """
        return self._meta['start_time']

    @property
    def device_id(self):
        return self._meta['device_id']
    
    @property
    def sync_recording(self):
        return self._sync_recording

    @property
    def stimulus(self):
        if self._meta.get('stimulus') is None:
            self._meta['stimulus'] = self.loader.load_stimulus(self)
        return self._meta.get('stimulus')

    def time_slice(self, start, stop):
        return RecordingView(self, start, stop)

    def __getitem__(self, chan):
        return self._channels[chan]

    def data(self):
        return np.stack([self[ch].data for ch in self.channels], axis=-1)

    @property
    def children(self):
        return [self[k] for k in self.channels]

    def __repr__(self):
        return f"<{self.__class__.__name__} device:{self.device_type}, channels:{str(self.channels)}>"

    def save(self):
        meta = self.meta.copy()
        if meta.get('stimulus') is not None:
            meta['stimulus'] = meta['stimulus'].save()
        channels = {k: v.save() for k, v in self._channels.items()}
        return {
            'schema version': (1, 0),
            'meta': meta,
            'start_time': self.start_time,
            'channels': channels,
        }

    @classmethod
    def load(cls, data):
        from neuroanalysis.stimuli import Stimulus

        meta = data['meta']
        if meta.get('stimulus') is not None:
            meta['stimulus'] = Stimulus.load(meta['stimulus'])
        channels = {k: TSeries.load(v) for k, v in data['channels'].items()}
        return cls(channels=channels, **meta)


class RecordingView(Recording):
    """A time-slice of a multi channel recording
    """
    def __init__(self, rec, start, stop):
        self._parent_rec = rec
        self._view_slice = (start, stop)
        chans = OrderedDict([(k, rec[k]) for k in rec.channels])
        meta = rec.meta.copy()
        Recording.__init__(self, channels=chans, sync_recording=rec.sync_recording, **meta)

    def __getattr__(self, attr):
        return getattr(self._parent_rec, attr)

    def __getitem__(self, item):
        return self._parent_rec[item].time_slice(*self._view_slice)

    @property
    def parent(self):
        return self._parent_rec

    # @property
    # def source_indices(self):
    #     """Return the indices of this view on the original Recording.
    #     """
    #     v = self
    #     start = 0
    #     while True:
    #         start += self._view_slice.start
    #         v = v.parent
    #         if not isinstance(v, RecordingView):
    #             break
    #     return start, start + len(self)


class PatchClampRecording(Recording):
    """Recording from a patch-clamp amplifier.

    * Current- or voltage-clamp mode
    * Minimum one recorded channel ('primary'), possibly more
    * Includes stimulus waveform ('command')
    * Stimulus metadata description
    * Metadata about amplifier state:
        * clamp_mode ('ic' 'i0', or 'vc')
        * holding potential (vc only)
        * holding_current (ic only)
        * bridge_balance (ic only)
        * lpf_cutoff
        * pipette_offset
    
    Should have at least 'primary' and 'command' channels.

    Note: command channel values should _include_ holding potential/current!

    Loader class needs methods:
        load_stimulus(self) - Return a neuroanalysis.stimuli.Stimulus object 
        load_test_pulse(self) - Return a neuroanalysis.test_pulse.PatchClampTestPulse object or None
        find_nearest_test_pulse(self) - Return the neuroanalysis.test_pulse.PatchClampTestPulse object recorded closest in time to this recording.
        get_baseline_regions(self) - Return a list of (start, stop) time pairs when we can expect this recording to be in a quiescent state.

    """
    def __init__(self, *args, **kwds):
        meta = OrderedDict()

        extra_meta = ['cell_id', 'clamp_mode', 'patch_mode', 'holding_potential', 'holding_current',
                      'bridge_balance', 'lpf_cutoff', 'pipette_offset', 'baseline_potential',
                      'baseline_current', 'baseline_rms_noise', 'stimulus']
                      
        for k in extra_meta:
            meta[k] = kwds.pop(k, None)

        Recording.__init__(self, *args, **kwds)
        self._meta.update(meta)

        self._baseline_regions = None
        self._baseline_data = None
        self._test_pulse = None
        self._nearest_test_pulse = None
        
    @property
    def cell_id(self):
        """Uniquely identifies the cell attached in this recording.
        """
        return self._meta['cell_id']
        
    @property
    def clamp_mode(self):
        """The mode of the patch clamp amplifier: 'vc', 'ic', or 'i0'.
        """
        return self._meta['clamp_mode'].lower()

    @property
    def patch_mode(self):
        """The state of the membrane patch. E.g. 'whole cell', 'cell attached', 'loose seal', 'bath', 'inside out', 'outside out'
        """
        return self._meta['patch_mode']

    @property
    def holding_potential(self):
        """The command holding potential if the recording is voltage-clamp, or the
        resting membrane potential if the recording is current-clamp.
        """
        if self.clamp_mode == 'vc':
            return self._meta['holding_potential']
        else:
            return self.baseline_potential
            
    @property        
    def rounded_holding_potential(self, increment=5e-3):
        """Return the holding potential rounded to the nearest increment.
        
        The default increment rounds to the nearest 5 mV.
        """
        hp = self.holding_potential
        if hp is None:
            return None
        return increment * np.round(hp / increment)

    @property
    def holding_current(self):
        """The steady-state pipette current applied during this recording.
        """
        if self.clamp_mode == 'ic':
            return self._meta['holding_current']
        else:
            return self.baseline_current

    @property
    def bridge_balance(self):
        """The bridge balance compensation applied during this recording.
        """
        return self._meta['bridge_balance']

    @property
    def test_pulse(self):
        if self._test_pulse is None:
            self._test_pulse = self.loader.load_test_pulse(self)
        return self._test_pulse  # may still be None

    @property
    def nearest_test_pulse(self):
        """The test pulse that was acquired nearest to this recording.
        """
        if self.test_pulse is not None:
            return self.test_pulse

        if self._nearest_test_pulse is None:
            self._nearest_test_pulse = self.loader.find_nearest_test_pulse(self)

        return self._nearest_test_pulse

    @property
    def baseline_regions(self):
        """A list of (start,stop) time pairs that cover regions of the recording
        the cell is expected to be in a steady state.
        """
        if self._baseline_regions is None:
            self._baseline_regions = self.loader.get_baseline_regions(self)
        return self._baseline_regions

    @property
    def baseline_data(self):
        """All items in baseline_regions concatentated into a single trace.
        """
        if self._baseline_data is None:
            data = [self['primary'].time_slice(start,stop).data for start,stop in self.baseline_regions]
            if len(data) == 0:
                data = np.empty(0, dtype=self['primary'].data.dtype)
            else:
                data = np.concatenate(data)
            data = data[np.isfinite(data)]
            self._baseline_data = TSeries(data, sample_rate=self['primary'].sample_rate, recording=self)
        return self._baseline_data

    @property
    def baseline_potential(self):
        """The mode potential value from all quiescent regions in the recording.

        See float_mode()
        """
        if self.meta['baseline_potential'] is None:
            if self.clamp_mode == 'vc':
                self.meta['baseline_potential'] = self.meta['holding_potential']
            else:
                data = self.baseline_data.data
                if len(data) == 0:
                    return None
                self.meta['baseline_potential'] = float_mode(data)
        return self.meta['baseline_potential']

    @property
    def baseline_current(self):
        """The mode current value from all quiescent regions in the recording.

        See float_mode()
        """
        if self.meta['baseline_current'] is None:
            if self.clamp_mode == 'ic':
                self.meta['baseline_current'] = self.meta['holding_current']
            else:
                data = self.baseline_data.data
                if len(data) == 0:
                    return None
                self.meta['baseline_current'] = float_mode(data)
        return self.meta['baseline_current']

    @property
    def baseline_rms_noise(self):
        """The standard deviation of all data from quiescent regions in the recording.
        """
        if self.meta['baseline_rms_noise'] is None:
            data = self.baseline_data.data
            if len(data) == 0:
                return None
            self.meta['baseline_rms_noise'] = data.std()
        return self.meta['baseline_rms_noise']

    def _descr(self):
        if self.clamp_mode == 'vc':
            hp = self.holding_potential
            if hp is not None:
                hp = int(np.round(hp * 1e3))
            return f"mode=VC holding={hp}"
        else:
            hc = self.holding_current
            if hc is not None:
                hc = int(np.round(hc * 1e12))
            return f"mode=IC holding={hc}"

    def __repr__(self):
        return f"<{self.__class__.__name__} device:{self.device_id} {self._descr()}>"


class TSeries(Container):
    """A homogeneous time series data set. 
    
    This is a representation of a single stream of data recorded over time. The
    data must be representable as a single N-dimensional array where the first
    array axis is time. 
    
    Examples:
    
    * A membrane potential recording from a single current-clamp headstage
    * A video stream from a camera
    * A digital trigger waveform
    
    TSeries may specify units, a starting time, and either a sample period / sample rate
    or an array of time values, one per sample.

    Parameters
    ----------
    data : array | None
        Array of data contained in this TSeries where the first axis is time.
    dt : float | None
        Optional value specifying the time difference between any two adjacent samples
        in the data; inverse of *sample_rate*. See ``TSeries.dt``.
    t0 : float | None
        Optional time value of the first sample in the data, relative to *start_time*. Default is 0.
        See ``TSeries.t0``.
    sample_rate : float | None
        Optional value specifying the sampling rate of the data; inverse of *dt*.
        See ``TSeries.sample_rate``.
    start_time : float | None
        Optional value giving the absloute starting time of the TSeries as a unix timestamp
        (seconds since epoch).  See ``TSeries.start_time``.
    time_values : array | None
        Optional array of the time values for each sample, relative to *start_time*. 
        This option can be used to specify data with timepoints that are irregularly sampled,
        and cannot be used with *dt*, *sample_rate*, or *t0*.
    units : str | None
        Optional string specifying the units associated with *data*. It is recommended
        to use unscaled SI units (e.g. 'V' instead of 'mV') where possible.
        See ``TSeries.units``.
    channel_id : str | None
        The name of the Recording channel that contains this TSeries
    recording : Recording | None
        The Recording this TSeries is part of.
    loader : a class | None
        A class containing functions for loading data. Required methods:
            get_tseries_data(self) - return a numpy array of values
    meta : 
        Any extra keyword arguments are interpreted as custom metadata and added to ``self.meta``.
    """
    def __init__(self, data: np.ndarray = None, dt=None, t0=None, sample_rate=None, start_time=None, time_values: np.ndarray = None, units=None, channel_id=None, recording=None, loader=None, **meta):
        Container.__init__(self, loader=loader)
        
        #if data is not None and data.ndim != 1:
        #    raise ValueError("data must be a 1-dimensional array.")
        
        if time_values is not None:
            if data is not None and time_values.shape[0] != data.shape[0]:
                raise ValueError("time_values must have the same length as data.")
            if dt is not None:
                raise TypeError("Cannot specify both time_values and dt.")
            if sample_rate is not None:
                raise TypeError("Cannot specify both time_values and sample_rate.")
            if t0 is not None:
                raise TypeError("Cannot specify both time_values and t0.")

        if dt is not None and sample_rate is not None:
            raise TypeError("Cannot specify both sample_rate and dt.")
            
        self._data = data
        self._meta = OrderedDict([
            ('start_time', start_time),
            ('dt', dt),
            ('t0', t0),
            ('sample_rate', sample_rate),
            ('units', units),
            ('channel_id', channel_id),
        ])
        self._meta.update(meta)
        self._time_values = time_values
        self._generated_time_values = None
        self._regularly_sampled = None
        self._recording = recording
        
    @property
    def data(self):
        """The array of sample values.
        """
        if self._data is None:
            self._data = self.loader.get_tseries_data(self)
        return self._data
        
    @property
    def start_time(self):
        """The clock time (seconds since epoch) corresponding to the sample
        where t=0. 
        
        If self.t0 is equal to 0, then start_time is the clock time of the
        first sample.
        """
        return self._meta['start_time']
    
    @property
    def sample_rate(self):
        """The sample rate for this TSeries.
        
        If no sample rate was specified, then this value is calculated from
        self.dt.
        """
        rate = self._meta['sample_rate']
        if rate is not None:
            return rate
        else:
            return 1.0 / self.dt

    @property
    def dt(self):
        """The time step between samples for this TSeries.
        
        If no time step was specified, then this value is calculated from
        self.sample_rate.
        
        If both dt and sample_rate were not specified, then this value
        is calculated as the difference between the first two items in
        time_values.
        
        If no timing information was specified at all, then accessing this
        property raises TypeError.
        """
        # need to be very careful about how we calculate dt and sample rate
        # to avoid fp errors.
        dt = self._meta['dt']
        if dt is not None:
            return dt
        
        rate = self._meta['sample_rate']
        if rate is not None:
            return 1.0 / rate
        
        t = self.time_values
        if t is not None:
            # assume regular sampling.
            # don't cache this value; we want to remember whether the user 
            # provided dt or samplerate
            return t[1] - t[0]
        
        raise TypeError("No sample timing is specified for this trace.")

    @property
    def t0(self):
        """The value of the first item in time_values.
        
        Setting this property causes the entire array of time values to shift.
        """
        t0 = self._meta['t0']
        if t0 is not None:
            return t0
        if self._time_values is not None:
            return self._time_values[0]
        return 0
    
    @t0.setter
    def t0(self, t0):
        if self.t0 == t0:
            return
        if self.has_time_values:
            self._time_values = self._time_values + (t0 - self._time_values[0])
        else:
            self._meta['t0'] = t0
            self._generated_time_values = None

    @property
    def t_end(self):
        """The last time value in this TSeries.
        """
        return self.time_at(len(self) - 1)

    def time_at(self, index):
        """Return the time at a specified index(es).

        Parameters
        ----------
        index : int | array-like
            The array index(es) for which time value(s) will be returned.
        """
        if not self.has_timing:
            raise TypeError("No sample timing is specified for this trace.")

        if not np.isscalar(index):
            index = np.asarray(index)

        if self.has_time_values:
            return self.time_values[index]
        else:
            # Be careful to minimize fp precision errors -- 
            #   time * dt != time / sample_rate != time * (1 / sample_rate)
            sample_rate = self._meta.get('sample_rate')
            if sample_rate is None:
                return (index * self.dt) + self.t0
            else:
                return (index * (1.0 / sample_rate)) + self.t0
 
    def index_at(self, t, index_mode=None):
        """Return the index at specified timepoint(s).

        Parameters
        ----------
        t : float | array-like
            The time value(s) for which array index(es) will be returned.
        index_mode : str
            Integer conversion mode: 'round' (default), 'floor', or 'ceil'. This argument is ignored
            if self.has_time_values is True.
        """
        if not self.has_timing:
            raise TypeError("No sample timing is specified for this trace.")

        if not np.isscalar(t):
            t = np.asarray(t)

        if self.has_time_values:
            inds1 = np.searchsorted(self.time_values, t)
            inds0 = inds1 - 1
            # select closest sample
            dif1 = abs(self.time_values[np.clip(inds1, 0, len(self)-1)] - t)
            dif0 = abs(self.time_values[inds0] - t)
            inds = np.where(dif0 < dif1, inds0, inds1)
            if np.isscalar(t):
                inds = int(inds)
        else:
            # Be careful to avoid fp precision errors when converting back to integer index
            sample_rate = self._meta.get('sample_rate')
            if sample_rate is None:
                inds = (t - self.t0) * (1.0 / self.dt)
            else:
                inds = (t - self.t0) * sample_rate
            
            if index_mode is None or index_mode == 'round':
                inds = np.round(inds)
            elif index_mode == 'floor':
                inds = np.floor(inds)
            elif index_mode == 'ceil':
                inds = np.ceil(inds)
            else:
                raise ValueError("index_mode must be 'round', 'ceil', or 'floor'; got %r" % index_mode)

            if np.isscalar(t):
                inds = int(inds)
            else:
                inds = inds.astype(int)
        return min(max(0, inds), len(self) - 1)

    @property
    def time_values(self):
        """An array of sample time values.
        
        Time values are specified in seconds relative to start_time.
        
        If no sample time values were provided for this TSeries, then the array
        is automatically generated based on other timing metadata (t0, dt,
        sample_rate).
        
        If no timing information at all was specified for this TSeries, then
        accessing this property raises TypeError.
        """
        if not self.has_timing:
            raise TypeError("No sample timing is specified for this trace.")

        if self.has_time_values:
            return self._time_values
        
        if self._generated_time_values is None:
            self._generated_time_values = self.time_at(np.arange(len(self.data)))
        
        return self._generated_time_values

    @property
    def regularly_sampled(self):
        """Boolean indicating whether the samples in this TSeries have equal
        time intervals.
        
        If either dt or sample_rate was specified for this trace, then this
        property is True. If only time values were given, then this property
        is True if the intervals between samples differ by less than 1%.
        
        If no sample timing was specified for this TSeries, then this property
        is False.
        """
        if not self.has_timing:
            return False
        
        if not self.has_time_values:
            return True
        
        if self._regularly_sampled is None:
            tvals = self.time_values
            dt = np.diff(tvals)
            avg_dt = dt.mean()
            self._regularly_sampled = bool(np.all(np.abs(dt - avg_dt) < (avg_dt * 0.01)))
        return self._regularly_sampled

    @property
    def has_timing(self):
        """Boolean indicating whether any timing information was specified for
        this TSeries.
        """
        return (self.has_time_values or 
                self._meta['dt'] is not None or 
                self._meta['sample_rate'] is not None)

    @property
    def has_time_values(self):
        """Boolean indicating whether an array of time values was explicitly
        specified for this TSeries.
        """
        return self._time_values is not None

    def time_slice(self, start, stop, index_mode=None):
        """Return a view of this trace with a specified start/stop time.
        
        Times are given relative to t0, and may be None to specify the
        beginning or end of the trace.
        
        Parameters
        ----------
        start : float
            Time at the start of the slice
        stop : float
            Time at the end of the slice (non-inclusive)
        index_mode : str
            See index_at for a description of this parameter.
        """
        i1 = max(0, self.index_at(start, index_mode)) if start is not None else None
        i2 = max(0, self.index_at(stop, index_mode)) if stop is not None else None
        return self[i1:i2]

    def value_at(self, t, interp='linear'):
        """Return the value of this trace at specific timepoints.

        By default, values are linearly interpolated from the data array.

        Parameters
        ----------
        t : float | array-like
            The time value(s) at which data value(s) will be returned.
        interp : 'linear' | 'nearest'
            If 'linear', then ``numpy.interp`` is used to interpolate values between adjacent samples.
            If 'nearest', then the sample nearest to each time value is returned.
            Default is 'linear'.
        """
        if not np.isscalar(t):
            t = np.asarray(t)

        if interp == 'linear':
            return np.interp(t, self.time_values, self.data)
        elif interp == 'nearest':
            inds = self.index_at(t)
            return self.data[inds]
        else:
            raise ValueError('unknown interpolation mode "%s"' % interp)

    @property
    def units(self):
        """Units string for the data in this TSeries.
        """
        return self._meta['units']

    @property
    def shape(self):
        """The shape of the array stored in this TSeries.
        """
        return self.data.shape
    
    def __len__(self):
        return self.shape[0]
    
    @property
    def duration(self):
        """Duration of this TSeries in seconds.

        If time values are specified for this trace, then this property
        is the difference between the first and last time values. 

        If only a sample rate or dt are specified, then this returns
        ``len(self) * dt``. 
        """
        if self.has_time_values:
            return self.time_values[-1] - self.t0
        else:
            return len(self) * self.dt

    @property
    def ndim(self):
        """Number of dimensions of the array contained in this TSeries.
        """
        return self.data.ndim
    
    @property
    def channel_id(self):
        """The name of the Recording channel that contains this TSeries.

        For example::

            trace = my_recording['primary']
            trace.recording   # returns my_recording
            trace.channel_id  # returns 'primary'
        """
        return self._meta['channel_id']
    
    @property
    def recording(self):
        """The Recording that contains this trace.
        """
        return self._recording

    @recording.setter
    def recording(self, new_val):
        self._recording = new_val

    def copy(self, data=None, time_values=None, **kwds):
        """Return a copy of this TSeries.
        
        The new TSeries will have the same data, timing information, and metadata
        unless otherwise specified in the arguments.
        
        Parameters
        ----------
        data : array | None
            If specified, sets the data array for the new TSeries.
        time_values : array | None
            If specified, sets the time_values array for the new TSeries.
        kwds :
            All extra keyword arguments will overwrite metadata properties.
            These include dt, sample_rate, t0, start_time, units, and
            others.
        """
        data, meta, tval = self._prepare_data_for_export(data, time_values, **kwds)

        return TSeries(data, time_values=tval, recording=self.recording, **meta)

    def _prepare_data_for_export(self, data=None, time_values=None, **kwds):
        if data is None:
            data = self.data.copy()
        if time_values is None:
            tval = self._time_values
            if tval is not None:
                tval = tval.copy()
        else:
            tval = time_values
        meta = self._meta.copy()
        meta.update(kwds)
        return data, meta, tval

    @property
    def parent(self):
        return self.recording

    def __getitem__(self, item):
        if isinstance(item, slice):
            return TSeriesView(self, item)
        else:
            raise TypeError("Invalid TSeries slice: %r" % item)

    def downsample(self, n=None, f=None):
        """Return a downsampled copy of this trace.
        
        Parameters
        ----------
        n : int
            (optional) number of samples to average
        f : float
            (optional) desired target sample rate
        """
        if not self.regularly_sampled:
            raise TypeError("downsample requires regularly-sampled data.")
        
        # choose downsampling factor
        if None not in (f, n):
            raise TypeError("Must specify either n or f (not both).")
        if n is None:
            if f is None:
                raise TypeError("Must specify either n or f.")
            n = int(np.round(self.sample_rate / f))
            if abs(n - (self.sample_rate / f)) > 1e-6:
                raise ValueError("Cannot downsample to %gHz; the resulting downsample factor is not an integer (try TSeries.resample instead)." % f)
        if n == 1:
            return self
        if n <= 0:
            raise Exception("Invalid downsampling factor: %d" % n)
        
        # downsample
        data = downsample(self.data, n, axis=0)
        
        # handle timing
        tvals = self._time_values
        if tvals is not None:
            tvals = tvals[::n]
        dt = self._meta['dt']
        if dt is not None:
            dt = dt * n
        sr = self._meta['sample_rate']
        if sr is not None:
            sr = float(sr) / n
        
        return self.copy(data=data, time_values=tvals, dt=dt, sample_rate=sr)

    def resample(self, sample_rate):
        """Return a resampled copy of this trace.
        
        Parameters
        ----------
        sample_rate : float
            The new sample rate of the returned TSeries

        Notes
        -----
        Lowpass filter followed by linear interpolation to extract new samples.

        Uses a bessel filter to avoid ringing artifacts, with cutoff=sample_rate
        and order=2 chosen to yield decent antialiasing and minimal blurring.

        scipy.resample was avoided due to ringing and edge artifacts.
        """
        if self.sample_rate == sample_rate:
            return self
        if not self.regularly_sampled:
            raise TypeError("resample requires regularly-sampled data.")
        
        ns = int(np.round(len(self) * sample_rate / self.sample_rate))

        # scipy.resample causes ringing and edge artifacts (freq-domain windowing
        # did not seem to help)
        # data = scipy.signal.resample(self.data, ns)

        # bessel filter gives reasonably good antialiasing with no ringing or edge
        # artifacts
        from ..filter import bessel_filter
        filt = bessel_filter(self, cutoff=sample_rate, order=2)
        t1 = self.time_values
        t2 = np.arange(t1[0], t1[-1], 1.0/sample_rate)

        data = np.interp(t2, t1, filt.data)
        
        if self._meta['sample_rate'] is not None:
            return self.copy(data=data, sample_rate=sample_rate)
        elif self._meta['dt'] is not None:
            dt = self.dt * self.sample_rate / sample_rate
            return self.copy(data=data, dt=dt)

    def __mul__(self, x):
        return self.copy(data=self.data * x)

    def __truediv__(self, x):
        return self.copy(data=self.data / x)

    def __add__(self, x):
        if isinstance(x, TSeries):
            x = x.data
        return self.copy(data=self.data + x)

    def __sub__(self, x):
        if isinstance(x, TSeries):
            x = x.data
        return self.copy(data=self.data - x)

    def concat(self, other):
        data = np.concatenate([self.data, other.data])
        times = np.concatenate([self.time_values, other.time_values])
        return self.copy(data=data, time_values=times)

    def mean(self):
        """Return the mean value of the data in this TSeries.

        Equivalent to self.data.mean()
        """
        return self.data.mean()

    def std(self):
        """Return the standard deviation of the data in this TSeries.

        Equivalent to self.data.std()
        """
        return self.data.std()

    def median(self):
        """Return the median value of the data in this TSeries.

        Equivalent to np.median(self.data)
        """
        return np.median(self.data)
        
    def diff(self):
        """Return the derivative of values in this trace with respect to its timebase.
        
        Note that the returned trace values are sampled half way between the time
        values of the input trace.
        """
        diff = np.diff(self.data)
        if self.has_time_values:
            dt = np.diff(self.time_values)
            t = self.time_values[:-1] + (0.5 * dt)
            dvdt = diff / dt
            return TSeries(data=dvdt, time_values=t)
        else:
            t0 = self.t0 + (0.5*self.dt)
            dvdt = diff / self.dt
            return TSeries(data=dvdt, t0=t0, dt=self.meta['dt'], sample_rate=self.meta['sample_rate'])
    
    def __repr__(self):
        if self.has_timing:
            if self.has_time_values:
                timing = "[has time values]"
            else:
                sample_rate = self._meta.get('sample_rate')
                if sample_rate is None:
                    timing = "t0=%g dt=%g" % (self.t0, self.dt)
                else:
                    timing = "t0=%g sample_rate=%g" % (self.t0, self.sample_rate)
        else:
            timing = "[no timing]"
            
        units = " units=" + self.units if self.units is not None else ""
            
        return "<%s length=%s %s%s>" % (
            self.__class__.__name__,
            len(self),
            timing,
            units,
        )

    def save(self):
        data, meta, tval = self._prepare_data_for_export()

        return {
            'schema version': (1, 0),
            'data': data,
            'time_values': tval,
            'meta': meta,
        }

    @classmethod
    def load(cls, data):
        meta = data.get('meta', {})
        time_values = data.get('time_values')
        data = data['data']
        return cls(data, time_values=time_values, **meta)


# for backward compatibility
Trace = TSeries


class TSeriesView(TSeries):
    def __init__(self, trace, sl):
        self._parent_trace = trace
        self._view_slice = sl
        inds = sl.indices(len(trace))
        self._view_indices = inds
        data = trace.data[sl]
        meta = trace.meta.copy()
        if trace.has_time_values:
            meta['time_values'] = trace.time_values[sl].copy()
        elif trace.has_timing:
            meta['t0'] = trace.time_at(inds[0])
        TSeries.__init__(self, data, recording=trace.recording, **meta)

    @property
    def parent(self):
        return self.source_trace.parent

    @property
    def recording(self):
        """The Recording that contains this trace.
        """
        return self.source_trace.recording

    @property
    def source_trace(self):
        """The original trace that is viewed by this TSeriesView.
        """
        v = self
        while True:
            v = v._parent_trace
            if not isinstance(v, TSeriesView):
                break
        return v

    @property
    def source_indices(self):
        """The indices of this view on the original TSeries.
        """
        v = self
        start = 0
        while True:
            start += self._view_slice.start
            v = v._parent_trace
            if not isinstance(v, TSeriesView):
                break
        return start, start + len(self)


class TSeriesList(object):
    def __init__(self, traces=None):
        self.traces = []
        if traces is not None:
            self.extend(traces)
            
    def __len__(self):
        return len(self.traces)
    
    def __iter__(self):
        return self.traces.__iter__()
    
    def __getitem__(self, i):
        return self.traces.__getitem__(i)
    
    def append(self, trace):
        self.traces.append(trace)
        
    def extend(self, traces):
        self.traces.extend(traces)
        
    def mean(self):
        """Return a trace with data averaged from all traces in this group.

        Downsamples to the minimum rate and clips ragged edges. All traces are aligned
        based on their _time values_ before being clipped and averaged. 
        """
        # Downsample all traces to the minimum sample rate
        min_sr = min([trace.sample_rate for trace in self.traces])
        downsampled = [trace.resample(min_sr) for trace in self.traces]
        
        # left-clip all traces to start at the same time
        start_t = max([trace.t0 for trace in downsampled])
        start_inds = [(trace, trace.index_at(start_t)) for trace in downsampled]
        
        # right-clip all traces to the same length
        clip_len = min([len(trace) - start_ind for trace, start_ind in start_inds])
        
        # stack together all cropped arrays
        clipped_data = [trace.data[start_ind:start_ind+clip_len] for trace, start_ind in start_inds]
        
        # average all traces together
        avg = np.nanmean(np.vstack(clipped_data), axis=0)
        
        # return a trace with the average data and all other properties taken from the
        # first downsampled trace
        ds = downsampled[0].copy(data=avg, t0=start_t, sample_rate=min_sr, dt=None)
        
        # Add metadata
        ds.meta['mean_of_n'] = len(clipped_data)
        
        return ds


class DAQRecording(Recording):
    """Input from / output to multiple channels on a data acquisition device.

    * Multiple channels of synchronized data (for each time point, one sample exists on every channel)
    * Metadata about DAQ state (sample rate, resolution, limits, filtering, etc)
    """


class ImageRecording(Recording):
    """Recording of one or more frames from a 2D imaging device (camera, 2p, confocal, etc).
    """


class VolumeRecording(Recording):
    """Recording of one or more frames from a 3D imaging device (MRI, lightfield, tomography, etc).
    """




