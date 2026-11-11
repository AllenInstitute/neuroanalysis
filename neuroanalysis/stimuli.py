import numbers

import numpy as np
import scipy.ndimage
import scipy.signal
from .util.custom_weakref import WeakRef
from .data import TSeries
from .fitting import psp


def _region_index(trace, t, index_mode=None):
    """Index of time *t* in *trace*, as TSeries.index_at, but clipped to [0, len(trace)] instead of
    [0, len(trace) - 1]: returns len(trace) if *t* falls past the last sample.
    """
    n = len(trace)
    if n == 0:
        return 0
    if trace.has_time_values:
        tv = trace.time_values
        # past the end if nearer to a virtual sample one interval after the last
        half_spacing = (tv[-1] - tv[-2]) / 2 if n > 1 else 0
        if t - tv[-1] > half_spacing:
            return n
        return trace.index_at(t)
    inds = (t - trace.t0) * trace.sample_rate
    if index_mode is None or index_mode == 'round':
        inds = np.round(inds)
    elif index_mode == 'floor':
        inds = np.floor(inds)
    elif index_mode == 'ceil':
        inds = np.ceil(inds)
    else:
        raise ValueError("index_mode must be 'round', 'ceil', or 'floor'; got %r" % index_mode)
    return int(min(max(0, inds), n))


def _region(trace, start, stop, index_mode=None):
    """Return a view of *trace* for times [start, stop).

    A region running past the end of the trace includes the last sample; a region entirely
    outside the trace is empty.
    """
    return trace[_region_index(trace, start, index_mode):_region_index(trace, stop, index_mode)]


def _eval_sample_rate(kwds):
    """Return the lowest sample rate (Hz) implied by Stimulus.eval() arguments, or None if unknown.
    """
    trace = kwds.get('trace')
    time_values = kwds.get('time_values')
    if trace is not None:
        if trace.has_time_values:
            time_values = trace.time_values
        else:
            return trace.sample_rate
    if time_values is not None:
        if len(time_values) < 2:
            return None
        return 1.0 / np.max(np.diff(time_values))
    if kwds.get('sample_rate') is not None:
        return kwds['sample_rate']
    if kwds.get('dt') is not None:
        return 1.0 / kwds['dt']
    return None


# Version of the saved stimulus state: the set of stimulus classes, their parameters, and their
# eval output. Bump it on any change to any of these. States without a version are legacy (0).
STIMULUS_SCHEMA_VERSION = 1


def load_stimulus(state):
    """Re-create a Stimulus structure from a previously saved state.

    States are generated using Stimulus.save().
    """
    return Stimulus.load(state)


class Stimulus(object):
    """Base metadata class for describing a stimulus (current injection, laser modulation, etc.)

    Stimulus descriptions are built as a hierarchy of Stimulus instances, where each item in
    the hierarchy may have multiple children that describe its sub-components. Stimulus
    subclasses each define a set of metadata fields and an optional eval() method that
    can be used to generate the stimulus.

    Parameters
    ----------
    description : str
        Human-readable description of this stimulus
    start_time : float
        The starting time of this stimulus relative to its parent's start_time.
    items : list | None
        An optional list of child Stimulus instances. 
    parent : Stimulus | None
        An optional parent Stimulus instance.

    Subclasses may declare named inputs in ``_inputs``: Stimulus-valued attributes that are
    evaluated separately and used by the host's eval, rather than summed into it. Inputs are not
    in ``items``; an input's ``start_time`` is relative to its host's global start time.


    Examples
    --------

    A waveform with two square pulses::

        stimulus = Stimulus(items=[
            SquarePulse(start_time=0.01, duration=0.01, amplitude=-50e-12),
            SquarePulse(start_time=0.2, duration=0.5, amplitude=200e-12),
        ], units='A')

    A waveform with a square pulse followed by a pulse train::

        stimulus = Stimulus(items=[
            SquarePulse(start_time=0.01, duration=0.01, amplitude=-50e-12),
            SquarePulseTrain(start_time=0.2, n_pulses=8, pulse_duration=0.002, amplitude=1.6e-9, interval=0.02),
        ], units='A')

    """
    _subclasses = {}

    _attributes = ['description', 'start_time', 'units']
    _inputs = []

    # subclasses may override with an instance attribute or property
    duration = 0

    def __init__(self, description="stimulus", start_time=0, units=None, items=None, parent=None):
        self.description = description
        self._start_time = start_time
        self.units = units

        self._items = []
        self._parent = WeakRef(None)
        self._owner = WeakRef(None)
        self.parent = parent

        for item in (items or []):
            self.append_item(item)        

    @property
    def type(self):
        """String type of this stimulus.
        """
        return type(self).__name__

    @property
    def parent(self):
        """The parent stimulus object, or None if there is no parent.
        """
        return self._parent()

    @parent.setter
    def parent(self, new_parent):
        old_parent = self.parent
        if old_parent is new_parent:
            return
        if new_parent is not None and self.owner is not None:
            raise ValueError("a stimulus used as an input cannot also be a child item")
        self._parent = WeakRef(new_parent)
        if old_parent is not None:
            try:
                old_parent.remove_item(self)
            except ValueError:
                pass  # already removed
        if new_parent is not None and not any(item is self for item in new_parent._items):
            new_parent.append_item(self)

    @property
    def owner(self):
        """The stimulus that uses this one as a named input, or None.
        """
        return self._owner()

    @property
    def inputs(self):
        """Dict of this stimulus's named inputs.
        """
        return {name: getattr(self, name) for name in self._inputs}

    def __setattr__(self, name, value):
        if name in self._inputs:
            self._attach_input(name, value)
        object.__setattr__(self, name, value)

    def _attach_input(self, name, stim):
        if not isinstance(stim, Stimulus):
            raise TypeError("input %r must be a Stimulus instance" % name)
        if stim.parent is not None or (stim.owner is not None and stim.owner is not self):
            raise ValueError("input %r already has a parent or owner" % name)
        old = self.__dict__.get(name)
        if old is not None and old is not stim:
            old._owner = WeakRef(None)
        stim._owner = WeakRef(self)

    def _eval_input(self, name, trace, index_mode=None):
        """Evaluate input *name* on a zero-filled copy of *trace*'s timebase; return its data array.
        """
        blank = trace.copy(data=np.zeros(len(trace)))
        return getattr(self, name).eval(trace=blank, index_mode=index_mode).data

    @property
    def items(self):
        """Tuple of child items contained within this stimulus.
        """
        return tuple(self._items)

    def append_item(self, item):
        """Append an item to the list of child stimuli.

        The item's parent will be set to this Stimulus.
        """
        self._items.append(item)
        item.parent = self

    def remove_item(self, item):
        """Remove an item from the list of child stimuli.

        The item's parent will be set to None. Items are matched by identity, not equality.
        """
        index = next((i for i, x in enumerate(self._items) if x is item), None)
        if index is None:
            raise ValueError("item is not a child of this stimulus")
        del self._items[index]
        item.parent = None

    def insert_item(self, index, item):
        """Insert an item into the list of child stimuli.

        The item's parent will be set to this Stimulus.
        """
        self._items.insert(index, item)
        item.parent = self        

    @property
    def global_start_time(self):
        """The global starting time of this stimulus.
        
        This is the sum of ``start_time`` along the chain of parents (or, for an input, its owner).
        """
        t = 0
        item = self
        while item is not None:
            t += item.start_time
            item = item.parent if item.parent is not None else item.owner
        return t

    @property
    def start_time(self):
        """The starting time of this stimulus relative to its parent's start time.
        """
        return self._start_time

    @property
    def global_end_time(self):
        """The global end time of this stimulus (global_start_time + duration).
        """
        return self.global_start_time + self.duration

    @property
    def end_time(self):
        """The ending time of this stimulus relative to its parent's start time.
        """
        return self.start_time + self.duration

    @property
    def total_duration(self):
        """The total duration of this stimulus and all of its children.
        """
        return self.total_global_end_time - self.global_start_time

    @property
    def total_global_end_time(self):
        """The global end time of this stimulus and all of its children.

        This is the value to use if you want to know when the stimulus is completely finished.
        """
        return max([self.global_end_time] + [item.total_global_end_time for item in self.items])

    @property
    def ancestry(self):
        """A generator yielding this item, its parent, and all grandparents.
        """
        item = self
        while item is not None:
            yield item
            item = item.parent

    def eval(self, time_values=None, n_pts=None, dt=None, sample_rate=None, t0=None, trace=None, index_mode='round'):
        """Return the value of this stimulus (as a TSeries instance) at defined timepoints.

        The result is generated by summing the eval results for all child items.

        The time points at which to evaluate the stimulus may be specified one of
        three ways:

        * an array of time values 
        * a set of arguments (t0, n_pts, dt/sample_rate) that can be converted to a TSeries
        * a TSeries instance

        Parameters
        ----------
        time_values : array | None
            An array of time values at which to evaluate the stimulus. This argument conflicts
            with *n_pts* and *trace*.
        n_pts : int | None
            The length of the TSeries to generate. This argument requires either *dt* or 
            *sample_rate* to be given as well.
        dt : float | None
            Timestep between samples in the generated TSeries (see TSeries.dt). This argument
            requires *n_pts* to be given as well, and conflits with *sample_rate*.
        sample_rate : float | None
            Sampling rate of the generated TSeries (see TSeries.sample_rate). This argument
            requires *n_pts* to be given as well, and conflits with *dt*.
        t0 : float | None
            The first timepoint to evaluate (see TSeries.t0). This argument requires *n_pts* and 
            either *dt* or *sample_rate* to be given as well.
        trace : TSeries | None
            If given, then the evaluated result is *added* to the data array contained in the TSeries.
        index_mode : str
            Mode used for converting float time values to integer indices ('round', 'floor', or 'ceil').
            See TSeries.index_at().
        """
        if time_values is None and t0 is None:
            t0 = 0
        trace = self._make_eval_trace(trace=trace, t0=t0, n_pts=n_pts, dt=dt, sample_rate=sample_rate, time_values=time_values)
        for item in self.items:
            item.eval(trace=trace, index_mode=index_mode)
        return trace

    def mask(self, trace=None, t0=None, n_pts=None, dt=None, sample_rate=None, time_values=None, index_mode='round'):
        """Return a TSeries that contains boolean data indicating the regions of the trace
        that would be affected by this stimulus.

        Arguments are the same as for ``Stimulus.eval()``.
        """
        trace = self._make_eval_trace(trace=trace, t0=t0, n_pts=n_pts, dt=dt, sample_rate=sample_rate, time_values=time_values)
        for item in self.items:
            item.mask(trace=trace, index_mode=index_mode)
        return trace

    def _make_eval_trace(self, trace=None, t0=None, n_pts=None, dt=None, sample_rate=None, time_values=None):
        if trace is not None:
            return trace
        if time_values is not None:
            data = np.zeros(len(time_values))
        else:
            assert n_pts is not None, "Must specify n_pts, time_values, or trace."
            data = np.zeros(n_pts)
        return TSeries(data, t0=t0, dt=dt, sample_rate=sample_rate, time_values=time_values, units=self.units)

    def __repr__(self):
        return '<{class_name} "{desc}" 0x{id:x}>'.format(class_name=type(self).__name__, desc=self.description, id=id(self))

    def __eq__(self, other):
        if not isinstance(other, Stimulus):
            return NotImplemented
        if self.type != other.type:
            return False
        if len(self.items) != len(other.items):
            return False
        for name in self._attributes:
            if self._save_value(getattr(self, name)) != self._save_value(getattr(other, name)):
                return False
        for i in range(len(self.items)):
            if self.items[i] != other.items[i]:
                return False
        for name in self._inputs:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def save(self):
        """Return a JSON-serializable representation of this Stimulus, its children, and inputs.

        The root state carries ``schema_version`` (see STIMULUS_SCHEMA_VERSION).
        """
        state = {'schema_version': STIMULUS_SCHEMA_VERSION}
        state.update(self._save_state())
        return state

    def _save_state(self):
        # unversioned state of this stimulus and its descendants; subclasses override this, not save()
        state = {
            'type': self.type,
            'args': {'start_time': float(self.start_time)},
        }
        for name in self._attributes:
            state['args'][name] = self._save_value(getattr(self, name))
        state['items'] = [item._save_state() for item in self.items]
        if self._inputs:
            state['inputs'] = {name: stim._save_state() for name, stim in self.inputs.items()}
        return state

    @classmethod
    def _save_value(cls, val):
        """Convert numpy scalars, arrays, and tuples to JSON-compatible Python types."""
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, (list, tuple)):
            return [cls._save_value(v) for v in val]
        return val

    @classmethod
    def load(cls, state):
        """Generate and return a Stimulus instance given a state data structure that was
        previously generated using Stimulus.save().

        States without ``schema_version`` are accepted as legacy; raises ValueError if the
        version is newer than STIMULUS_SCHEMA_VERSION.
        """
        version = state.get('schema_version', 0)
        if version > STIMULUS_SCHEMA_VERSION:
            raise ValueError(
                "Stimulus state has schema version %s; this version of neuroanalysis supports up to %d."
                % (version, STIMULUS_SCHEMA_VERSION))
        return cls._load_state(state)

    @classmethod
    def _load_state(cls, state):
        item_class = cls.get_stimulus_class(state['type'])
        inputs = {name: cls._load_state(s) for name, s in state.get('inputs', {}).items()}
        item = item_class(**state.get('args', {}), **inputs)
        for item_state in state.get('items', []):
            item.append_item(cls._load_state(item_state))
        return item

    @classmethod
    def get_stimulus_class(cls, name):
        """Return the Stimulus class or descendant (at any depth) named *name*."""
        registry = Stimulus._subclasses
        if name not in registry:
            registry.clear()
            todo = [Stimulus]
            while todo:
                sub = todo.pop()
                registry[sub.__name__] = sub
                todo.extend(sub.__subclasses__())
        if name not in registry:
            raise KeyError('Unknown stimulus class "%s"' % name)
        return registry[name]


class LazyLoadStimulus(Stimulus):
    """A Stimulus whose child items are loaded on first access via ``loader.load_stimulus_items(source)``.

    Loaded items are placed before any items added before loading. The loader is called once.
    """
    def __init__(self, description, start_time=0, units=None, items=None, parent=None, loader=None, source=None):
        if loader is None:
            raise Exception("Use of a LazyLoadStimulus requires a loader to be specified upon init.")
        if source is None:
            raise Exception("Use of a LazyLoadStimulus requires a source to be specified upon init.")

        self._loader = loader
        self._source = source
        self._loaded = False
        Stimulus.__init__(self, description, start_time=start_time, units=units, items=items, parent=parent)

    @property
    def items(self):
        if not self._loaded:
            loaded = self._loader.load_stimulus_items(self._source)
            self._loaded = True
            for i, item in enumerate(loaded):
                self.insert_item(i, item)
        return tuple(self._items)

    def _save_state(self):
        # saved as a plain Stimulus containing the loaded items; the loader is not serializable
        state = Stimulus._save_state(self)
        state['type'] = 'Stimulus'
        return state


class Offset(Stimulus):
    """A constant offset in the stimulus.

    This offset is applied at all timepoints in the stimulus that are after or equal to the start time.

    Parameters
    ----------
    amplitude : float
        The offset to apply.
    start_time : float
        The time point at which the offset begins.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['amplitude']

    def __init__(self, amplitude, start_time=0, description="offset", units=None, parent=None):
        self.amplitude = amplitude
        Stimulus.__init__(self, description=description, start_time=start_time, units=units, parent=parent)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        start_ind = _region_index(trace, self.global_start_time, index_mode=kwds.get('index_mode'))
        trace.data[start_ind:] += self.amplitude
        return trace

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start_ind = _region_index(trace, self.global_start_time, index_mode=kwds.get('index_mode'))
        trace.data[start_ind:] = True
        return trace
        

class SquarePulse(Stimulus):
    """A square pulse stimulus.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    duration : float
        The duration in seconds of the pulse.
    amplitude : float
        The amplitude of the pulse.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['duration', 'amplitude']

    def __init__(self, start_time, duration, amplitude, description="square pulse", units=None, parent=None):
        if duration < 0:
            raise ValueError("duration must be non-negative")
        self.duration = duration
        self.amplitude = amplitude
        Stimulus.__init__(self, description=description, start_time=start_time, units=units, parent=parent)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        _region(trace, start, start+self.duration, index_mode=kwds.get('index_mode')).data[:] += self.amplitude
        return trace

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        _region(trace, start, start+self.duration, index_mode=kwds.get('index_mode')).data[:] = True
        return trace


def find_square_pulses(trace, baseline=None) -> list[SquarePulse]:
    """Return a list of SquarePulse instances describing square pulses found
    in the stimulus.
    
    A pulse is defined as any contiguous region of the stimulus waveform
    that has a constant value other than the baseline. If no baseline is
    specified, then the first sample in the stimulus is used.
    
    Parameters
    ----------
    trace : TSeries instance
        The stimulus command waveform. This data should be noise-free and nan-free.
    baseline : float | None
        Specifies the value in the command waveform that is considered to be
        "no pulse". If no baseline is specified, then the first sample of
        *trace* is used.
    """
    if not isinstance(trace, TSeries):
        raise TypeError("argument must be TSeries instance")
    if baseline is None:
        baseline = trace.data[0]
    sdiff = np.diff(trace.data)
    changes = np.argwhere(sdiff != 0)[:, 0] + 1
    pulses = []
    for i, start in enumerate(changes):
        amp = trace.data[start] - baseline
        if amp != 0:
            stop = changes[i+1] if (i+1 < len(changes)) else len(trace)
            t_start = trace.time_at(start)
            duration = (stop - start) * trace.dt
            pulses.append(SquarePulse(start_time=t_start, duration=duration, amplitude=amp, units=trace.units))
            pulses[-1].pulse_number = i
    return pulses


def find_noisy_square_pulses(trace, baseline=None, std_threshold=5.0, min_duration=0, min_amplitude=0):
    """Return a list of SquarePulse instances describing square pulses found
    in the given trace.
    
    A pulse is defined as any contiguous region of the stimulus waveform
    that has a value outside the amp_threshold or std_threshold from the 
    baseline. If no baseline is specified, then the first 200 samples in the 
    stimulus are used.
    
    Parameters
    ----------
    trace : TSeries instance
        The stimulus waveform. This can contain noise - for noise free data 
        see `find_square_pulse`.
    baseline : numpy.array | None
        Specify an array to use as the baseline (a region considered to be 
        "no pulse"). If no baseline is specified, then the first 200 samples of
        *trace* are used.
    std_threshold: float | 3.0
        How many stdev's the pulse must be from the baseline to be detected. 
    min_duration: float | 0
        If specified, the minimum duration of a pulse (in seconds). Pulses shorter
        than min_duration will be discarded.
    min_amplitude: float | 0
        If specified, the minimum amplitude of a pulse (absolute value). Pulses 
        with absolute value amplitudes smaller than min_amplitude will be discarded.
    """
    if not isinstance(trace, TSeries):
        raise TypeError("argument must be TSeries instance")

    if baseline is None:
        baseline = trace.data[:200]

    threshold = baseline.std()*std_threshold

    sdiff = np.diff(trace.data - baseline.mean())
    changes = np.argwhere(abs(sdiff) > threshold)[:, 0]

    ### sometimes square pulses aren't quite square - only count the diff if the index before it is below threshold
    real_changes= []
    for c in changes:
        if (abs(sdiff[c-1]) < threshold) and (abs(sdiff[c]) > threshold):
            real_changes.append(c+1) ## add one to get the first index at the new value (the start of the pulse, rather than the last point of baseline)
    changes = real_changes

    pulses = []
    stop = 0
    for i, start in enumerate(changes):
        if len(pulses) > 0 and stop >= start: ## this is the end of a pulse
            continue
        #amp = trace.data[start] - baseline.mean()
        #if abs(amp) > threshold: ## should only be true at the start of pulses
        else:
            stop = changes[i+1] if (len(changes) > i+1) else len(trace)
            t_start = trace.time_at(start)
            duration = (stop - start) * trace.dt
            amplitude = trace.data[start:stop].mean() - baseline.mean()
            if duration > min_duration and abs(amplitude) > min_amplitude:
                pulses.append(SquarePulse(start_time=t_start, duration=duration, amplitude=amplitude, units=trace.units))

    return pulses


class SquarePulseTrain(Stimulus):
    """A train of identical, regularly-spaced square pulses.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    n_pulses : int
        The number of pulses in the train.
    pulse_duration : float
        The duration in seconds of a single pulse.
    amplitude : float
        The amplitude of a single pulse.
    interval : float
        The time in seconds between the onset of adjacent pulses.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['n_pulses', 'pulse_duration', 'amplitude', 'interval']

    def __init__(self, start_time, n_pulses, pulse_duration, amplitude, interval, description="square pulse train", units=None, parent=None):
        if not isinstance(n_pulses, numbers.Integral):
            raise TypeError("n_pulses must be an integer")
        n_pulses = int(n_pulses)
        if n_pulses < 1:
            raise ValueError("n_pulses must be a positive integer")
        if interval < pulse_duration:
            raise ValueError("interval must be >= pulse_duration (pulses would overlap)")
        self.n_pulses = n_pulses
        self.pulse_duration = pulse_duration
        self.amplitude = amplitude
        self.interval = interval
        Stimulus.__init__(self, description=description, start_time=start_time, units=units, parent=parent)

        pulse_times = np.arange(n_pulses) * interval
        for i,t in enumerate(pulse_times):
            pulse = SquarePulse(start_time=t, duration=pulse_duration, amplitude=amplitude, parent=self, units=units)
            pulse.pulse_number = i

    @property
    def global_pulse_times(self):
        """A list of the global start times of all pulses in the train.
        """
        return [t + self.global_start_time for t in self.pulse_times]

    @property
    def pulse_times(self):
        """A list of the start times of all pulses in the train.
        """
        return [item.start_time for item in self.items]

    def _save_state(self):
        state = Stimulus._save_state(self)
        state['items'] = []  # don't save auto-generated items
        return state
        

class SquarePulseSeries(Stimulus):
    """A series of square pulses of varying amplitude, duration, and timing.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    pulse_times : float array
        Array of starting times for each pulse relative to *start_time*.
    pulse_durations : float array
        Array of pulse durations in seconds.
    amplitudes : float array
        Array of pulse amplitudes.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['pulse_times', 'pulse_durations', 'amplitudes']

    def __init__(self, start_time, pulse_times, pulse_durations, amplitudes, description="square pulse series", units=None, parent=None):
        self.pulse_times = pulse_times
        self.pulse_durations = pulse_durations
        self.amplitudes = amplitudes
        if not (len(pulse_times) == len(pulse_durations) == len(amplitudes)):
            raise ValueError("pulse_times, pulse_durations, and amplitudes must all have the same length")
        Stimulus.__init__(self, description=description, start_time=start_time, units=units, parent=parent)

        for i,t in enumerate(pulse_times):
            pulse = SquarePulse(start_time=t, duration=pulse_durations[i], amplitude=amplitudes[i], parent=self, units=units)
            pulse.pulse_number = i

    @property
    def global_pulse_times(self):
        """A list of the global start times of all pulses in the train.
        """
        return [t + self.global_start_time for t in self.pulse_times]

    def _save_state(self):
        state = Stimulus._save_state(self)
        state['items'] = []  # don't save auto-generated items
        return state
        

class Ramp(Stimulus):
    """A linear ramp.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    duration : float
        The duration in seconds of the ramp.
    slope : float
        The slope of the ramp in units per second.
    offset : float
        A constant offset added to the stimulus during the ramp.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['duration', 'slope', 'offset']

    def __init__(self, start_time, duration, slope, offset=0, description="linear ramp", units=None, parent=None):
        if duration < 0:
            raise ValueError("duration must be non-negative")
        self.duration = duration
        self.slope = slope
        self.offset = offset
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        region = _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode'))
        region.data[:] += (region.time_values - start) * self.slope + self.offset
        return trace

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        region = _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode'))
        region.data[:] = True
        return trace


class Sine(Stimulus):
    """A sine wave.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    duration : float
        The duration in seconds of the sine wave.
    frequency : float
        The sine wave frequency (Hz).
    amplitude : float
        The peak amplitude of the sine wave.
    phase : float
        The initial phase of the sine wave at *start_time*.
    offset : float
        A constant offset added to the stimulus during the sine wave.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['duration', 'frequency', 'amplitude', 'phase', 'offset']

    def __init__(self, start_time, duration, frequency, amplitude, phase=0, offset=0, description="sine wave", units=None, parent=None):
        if duration < 0:
            raise ValueError("duration must be non-negative")
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.duration = duration
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.offset = offset
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    def eval(self, **kwds):
        sample_rate = _eval_sample_rate(kwds)
        if sample_rate is not None and self.frequency > sample_rate / 2:
            raise ValueError("Sample rate must be at least twice the frequency of the sine wave.")
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start+self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] += self.offset

        t = chunk.time_values - start
        phase = self.phase_at(t)
        chunk.data[:] += self.amplitude * np.sin(phase)

        return trace

    def phase_at(self, t):
        """Return the phase of the sine wave at time (or array of times) *t* relative
        to the start time.
        """
        return self.phase + (t * (2 * np.pi * self.frequency))

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start+self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] = True
        return trace


class Chirp(Stimulus):
    """A frequency-chirped sinusoid.

    The frequency of the chirp is swept in a geometric progression.

    Parameters
    ----------
    start_time : float
        The starting time of the first pulse in the train, relative to the start time of the parent
        stimulus.
    duration : float
        The duration in seconds of the chirp.
    start_frequency : float
        The initial sine wave frequency (Hz) at *start_time*.
    end_frequency : float
        The final sine wave frequency (Hz) at *start_time* + *duration*.
    amplitude : float
        The peak amplitude of the chirp.
    phase : float
        The initial phase of the sine wave at *start_time*.
    offset : float
        A constant offset added to the stimulus during the chirp.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.


    Notes
    -----

    The waveform calculation follows two main requirements:

    1. The frequency should sweep in a geometric progression
    2. The frequency at the start and end of the sweep are specified

    To satisfy these conditions, we start with the basic form for a sine wave::

        y(t) = sin(w(t))

    Since we know *w* must follow a geometric progression in time, we can give it a little more structure::

        w(t) = k r^(t/d)

    .. where k and r are constants for us to solve, and *d* is the duration of the chirp (adding *d* here simplifies the math later on).
    Take the derivative with respect to time::

        dw/dt = k ln(r) r^(t/d) / d

    Now we can apply the frequency constraints to solve for k and r::

        f(t) = dw/dt / (2 π)
        f(0) = k ln(r) / (2 π d)
        f(d) = k r ln(r) / (2 π d)

    Solving for k and r, we get::

        k = 2 π f(0) d / ln(r)
        r = f(d) / f(0)

    And the final equation (without phase and offset) is::

        y(t) = sin[2 π f(0) d (f(d) / f(0))^(t / d) / ln(f(d) / f(0)]
    """
    _attributes = Stimulus._attributes + ['duration', 'start_frequency', 'end_frequency', 'amplitude', 'phase', 'offset']

    def __init__(self, start_time, duration, start_frequency, end_frequency, amplitude, phase=0, offset=0, description="frequency chirp", units=None, parent=None):
        if duration <= 0:
            raise ValueError("duration must be positive")
        if start_frequency <= 0:
            raise ValueError("start_frequency must be positive")
        if end_frequency <= 0:
            raise ValueError("end_frequency must be positive")
        self.duration = duration
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.amplitude = amplitude
        self.phase = phase
        self.offset = offset
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    def eval(self, **kwds):
        sample_rate = _eval_sample_rate(kwds)
        if sample_rate is not None and max(self.start_frequency, self.end_frequency) > sample_rate / 2:
            raise ValueError("Sample rate must be at least twice the maximum frequency in the chirp.")
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] += self.offset

        t = chunk.time_values - start
        d2 = self.amplitude * np.sin(self.phase_at(t))

        chunk.data[:] += d2
        return trace

    def _kr(self):
        # return constants k and r (see mathy notes above)
        f0, f1 = self.start_frequency, self.end_frequency
        r = f1 / f0
        k = 2 * np.pi * self.duration * f0 / np.log(r)        
        return k, r

    def phase_at(self, t):
        """Return the phase of the chirp at time (or array of times) *t* relative
        to the start of the chirp.
        """
        if self.start_frequency == self.end_frequency:
            return self.phase + 2 * np.pi * self.start_frequency * t
        k, r = self._kr()
        w = k * r ** (t / self.duration)
        return self.phase - k + w

    def frequency_at(self, t):
        """Return the frequency of the chirp at time (or array of times) *t* relative
        to the start of the chirp.
        """
        # f(t) = k ln(r) r^(t/d) / (2 π d)
        if self.start_frequency == self.end_frequency:
            return self.start_frequency + 0 * np.asarray(t, dtype=float)
        k, r = self._kr()
        d = self.duration
        return k * np.log(r) * r ** (t/d) / (2 * np.pi * d)

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start+self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] = True
        return trace


class Psp(Stimulus):
    """A PSP- or PSC-shaped stimulus.

    This shape is the product of rising and decaying exponentials. See ``neuroanalysis.fitting.psp.Psp``.

    Parameters
    ----------
    start_time : float
        The starting time (s) of the stimulus.
    rise_time : float
        Time (s) from stimulus start until the peak of the PSP shape.
    decay_tau : float
        Exponential decay time constant (s).
    amplitude : float
        The peak amplitude of the PSP shape.
    rise_power : float
        Exponent modifying the rising exponential (default is 2; larger values yield a slower initial activation, 1 yields instantaneous activation).
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.

    """
    _attributes = Stimulus._attributes + ['rise_time', 'decay_tau', 'amplitude', 'rise_power']

    def __init__(self, start_time, rise_time, decay_tau, amplitude, rise_power=2, description="PSP", units=None, parent=None):
        if rise_time <= 0:
            raise ValueError("rise_time must be positive")
        if decay_tau <= 0:
            raise ValueError("decay_tau must be positive")
        if rise_power <= 0:
            raise ValueError("rise_power must be positive")
        self.rise_time = rise_time
        self.decay_tau = decay_tau
        self.amplitude = amplitude
        self.rise_power = rise_power
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    @property
    def duration(self):
        return 15 * max(self.rise_time, self.decay_tau)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] += psp.Psp.psp_func(
            x=chunk.time_values,
            xoffset=start,
            yoffset=0,
            rise_time=self.rise_time,
            decay_tau=self.decay_tau,
            amp=self.amplitude,
            rise_power=self.rise_power,
        )
        return trace

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start+self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] = True
        return trace


class Noise(Stimulus):
    """Gaussian noise with a power-law spectrum, over [start_time, start_time + duration].

    Generated by FFT spectral shaping of a seeded normal draw, then normalized so that the
    samples in the window have exactly the given mean and standard deviation. Assumes uniform
    sampling. The realization depends on the sample timebase: the same seed at a different
    sample rate or window length gives different noise.

    Parameters
    ----------
    start_time : float
        Start time (s) relative to the parent.
    duration : float
        Duration (s) of the noise.
    sigma : float
        Standard deviation of the noise over the window.
    mean : float
        Mean of the noise over the window.
    noise_type : str
        Power spectrum shape: 'white' (flat), 'pink' (1/f), 'brown' (1/f^2), 'blue' (f), or 'violet' (f^2).
    seed : int | None
        Random seed. If None, a seed is chosen at construction and saved with the stimulus.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['duration', 'sigma', 'mean', 'noise_type', 'seed']

    # power spectral density exponent for each noise type
    spectral_exponents = {'white': 0, 'pink': -1, 'brown': -2, 'blue': 1, 'violet': 2}

    def __init__(self, start_time, duration, sigma, mean=0, noise_type='white', seed=None, description="noise", units=None, parent=None):
        if duration < 0:
            raise ValueError("duration must be non-negative")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if noise_type not in self.spectral_exponents:
            raise ValueError("noise_type must be one of %s; got %r" % (list(self.spectral_exponents), noise_type))
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % 2**63)
        self.duration = duration
        self.sigma = sigma
        self.mean = mean
        self.noise_type = noise_type
        self.seed = seed
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    def eval(self, **kwds):
        trace = Stimulus.eval(self, **kwds)
        start = self.global_start_time
        chunk = _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode'))
        chunk.data[:] += self.noise_values(len(chunk))
        return trace

    def noise_values(self, n):
        """Return *n* samples of this noise (the values eval adds over its window).
        """
        if n == 0:
            return np.zeros(0)
        x = np.random.default_rng(self.seed).standard_normal(n)
        exponent = self.spectral_exponents[self.noise_type]
        if exponent != 0:
            freqs = np.fft.rfftfreq(n)
            scale = np.zeros(len(freqs))
            scale[1:] = freqs[1:] ** (exponent / 2)
            x = np.fft.irfft(np.fft.rfft(x) * scale, n)
        x = x - x.mean()
        std = x.std()
        if std > 0:
            x *= self.sigma / std
        return x + self.mean

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode')).data[:] = True
        return trace


class Pwm(Stimulus):
    """Pulse-width modulation of a square wave by a duty-cycle input.

    Cycles start at start_time + k * period. Each cycle outputs *amplitude* for its first
    duty * period and 0 for the rest, where duty is the mean of the *duty* input over that cycle,
    clipped to [0, 1]. The final cycle is truncated at start_time + duration.

    Parameters
    ----------
    start_time : float
        Start time (s) relative to the parent.
    duration : float
        Duration (s) of the PWM output.
    period : float
        Duration (s) of one PWM cycle.
    amplitude : float
        Output value during the "on" part of each cycle.
    duty : Stimulus
        Input giving the duty cycle (0-1); its start_time is relative to this stimulus's start.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['duration', 'period', 'amplitude']
    _inputs = ['duty']

    def __init__(self, start_time, duration, period, amplitude, duty, description="pwm", units=None, parent=None):
        if duration < 0:
            raise ValueError("duration must be non-negative")
        if period <= 0:
            raise ValueError("period must be positive")
        self.duration = duration
        self.period = period
        self.amplitude = amplitude
        self.duty = duty
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    def eval(self, **kwds):
        index_mode = kwds.get('index_mode')
        trace = Stimulus.eval(self, **kwds)
        duty = self._eval_input('duty', trace, index_mode=index_mode)
        start = self.global_start_time
        stop = start + self.duration
        n_cycles = int(np.ceil(round(self.duration / self.period, 9)))
        for k in range(n_cycles):
            cycle_start = start + k * self.period
            cycle_stop = min(cycle_start + self.period, stop)
            i0 = _region_index(trace, cycle_start, index_mode=index_mode)
            i1 = _region_index(trace, cycle_stop, index_mode=index_mode)
            if i1 <= i0:
                continue
            frac = float(np.clip(duty[i0:i1].mean(), 0, 1))
            on_stop = min(cycle_start + frac * self.period, cycle_stop)
            _region(trace, cycle_start, on_stop, index_mode=index_mode).data[:] += self.amplitude
        return trace

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode')).data[:] = True
        return trace


class Filter(Stimulus):
    """A lowpass-filtered copy of an input stimulus.

    The input is evaluated and filtered over the entire evaluated trace, which must be regularly
    sampled. Bessel and Butterworth filters are causal (like a hardware filter); samples before
    the trace are taken to equal the first sample. The Gaussian filter is symmetric (zero-phase),
    with kernel sigma = sqrt(ln 2) / (2 * pi * cutoff) seconds, which puts its
    -3 dB point at *cutoff*. Filter ringing and tails may extend outside ``mask()``.

    Parameters
    ----------
    input : Stimulus
        The stimulus to filter; its start_time is relative to this stimulus's start.
    filter_type : str
        'bessel', 'butterworth', or 'gaussian'.
    cutoff : float
        -3 dB frequency (Hz).
    order : int
        Filter order for 'bessel' and 'butterworth'; ignored for 'gaussian'.
    start_time : float
        Start time (s) relative to the parent; shifts the input.
    description : str
        Optional string describing the stimulus.
    units : str | None
        Optional string describing the units of values in the stimulus.
    """
    _attributes = Stimulus._attributes + ['filter_type', 'cutoff', 'order']
    _inputs = ['input']

    filter_types = ('bessel', 'butterworth', 'gaussian')

    def __init__(self, input, filter_type, cutoff, order=4, start_time=0, description="filter", units=None, parent=None):
        if filter_type not in self.filter_types:
            raise ValueError("filter_type must be one of %s; got %r" % (list(self.filter_types), filter_type))
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("order must be a positive integer")
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.order = order
        self.input = input
        Stimulus.__init__(self, description=description, start_time=start_time, parent=parent, units=units)

    @property
    def duration(self):
        """Time (s) from this stimulus's start to the end of its input, including the input's children.
        """
        return max(0, self.input.total_global_end_time - self.global_start_time)

    def eval(self, **kwds):
        index_mode = kwds.get('index_mode')
        trace = Stimulus.eval(self, **kwds)
        if not trace.regularly_sampled:
            raise ValueError("Filter requires a regularly sampled trace")
        x = self._eval_input('input', trace, index_mode=index_mode)
        trace.data[:] += self.filter_values(x, trace.sample_rate)
        return trace

    def filter_values(self, x, sample_rate):
        """Return array *x* filtered by this filter at *sample_rate* (Hz).
        """
        if len(x) == 0:
            return x.copy()
        if self.filter_type == 'gaussian':
            sigma = np.sqrt(np.log(2)) / (2 * np.pi * self.cutoff)
            return scipy.ndimage.gaussian_filter1d(x, sigma * sample_rate, mode='nearest')
        if self.filter_type == 'bessel':
            sos = scipy.signal.bessel(self.order, self.cutoff, norm='mag', fs=sample_rate, output='sos')
        else:
            sos = scipy.signal.butter(self.order, self.cutoff, fs=sample_rate, output='sos')
        y, _ = scipy.signal.sosfilt(sos, x, zi=scipy.signal.sosfilt_zi(sos) * x[0])
        return y

    def mask(self, **kwds):
        trace = Stimulus.mask(self, **kwds)
        start = self.global_start_time
        _region(trace, start, start + self.duration, index_mode=kwds.get('index_mode')).data[:] = True
        return trace
