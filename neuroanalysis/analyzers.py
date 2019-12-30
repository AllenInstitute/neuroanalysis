


class Analyzer(object):
    """Base class for attaching analysis results to a data object.
    """
    @classmethod
    def get(cls, obj):
        """Get the analyzer attached to a recording, or create a new one.
        """
        analyzer = getattr(obj, '_' + cls.__name__, None)
        if analyzer is None:
            analyzer = cls(obj)
        return analyzer

    def _attach(self, obj):
        attr = '_' + self.__class__.__name__
        if hasattr(obj, attr):
            raise TypeError("Object %s already has attached %s" % (obj, self.__class__.__name__))
        setattr(obj, attr, self)


class BaselineAnalyzer(Analyzer):

    def __init__(self, rec):
        self._attach(rec)
        self.rec = rec

        self._baseline_regions = None 
        self._baseline_data = None
        self._baseline_potential = None
        self._baseline_current = None
        self._baseline_rms_noise = None

        self._baseline_chunk_list = None
        self._ptr = 0

    @property
    def baseline_regions(self):
        """A list of (start,stop) time pairs where the recording can be expected to be quiescent.
        """
        raise Exception("Must be reimplemented in subclass")


    def get_baseline_chunk(self, duration=20e-3):
        """Return the (start, stop) indices of a chunk of unused baseline with the
        given duration.
        """
        while True:
            if self._baseline_chunk_list == None:
                self._baseline_chunk_list = self.baseline_regions
            if len(self._baseline_chunk_list) == 0:
                return None
            start, stop = self._baseline_chunk_list[0]
            chunk_start = max(start, self._ptr)
            chunk_stop = chunk_start + duration
            if chunk_stop <= stop:
                self._ptr = chunk_stop
                return chunk_start, chunk_stop
            else:
                self._baseline_chunk_list.pop(0)

    @property
    def baseline_data(self):
        """All items in baseline_regions concatentated into a single trace.
        """
        if self._baseline_data is None:
            data = [self.rec['primary'].time_slice(start,stop).data for start,stop in self.baseline_regions]
            if len(data) == 0:
                data = np.empty(0, dtype=self.rec['primary'].data.dtype)
            else:
                data = np.concatenate(data)
            self._baseline_data = TSeries(data, sample_rate=self.rec['primary'].sample_rate, recording=self.rec)
        return self._baseline_data

    @property
    def baseline_potential(self):
        """The mode potential value from all quiescent regions in the recording.

        See float_mode()
        """
        if self._baseline_potential is None:
            if self.rec.clamp_mode == 'vc':
                self._baseline_potential = self.rec.meta['holding_potential']
            else:
                data = self.baseline_data.data
                if len(data) == 0:
                    return None
                self._baseline_potential = float_mode(data)
        return self._baseline_potential

    @property
    def baseline_current(self):
        """The mode current value from all quiescent regions in the recording.

        See float_mode()
        """
        if self._baseline_current is None:
            if self.rec.clamp_mode == 'ic':
                self._baseline_current = self.rec.meta['holding_current']
            else:
                data = self.baseline_data.data
                if len(data) == 0:
                    return None
                self._baseline_current = float_mode(data)
        return self._baseline_current

    @property
    def baseline_rms_noise(self):
        """The standard deviation of all data from quiescent regions in the recording.
        """
        if self._baseline_rms_noise is None:
            data = self.baseline_data.data
            if len(data) == 0:
                return None
            self._baseline_rms_noise = data.std()
        return self._baseline_rms_noise
