from neuroanalysis.analyzers.analyzer import Analyzer

class BaselineAnalyzer(Analyzer):

    _settle_time = None ## (float) The amount of time (in seconds) to allow the cell to settle back
                        ## to baseline after the end of a stimulus. 

    def __init__(self, sync_rec):
        self._attach(sync_rec)
        self.sync_rec = sync_rec

        self._baseline_regions = None

    @property
    def settle_time(self):
        if self._settle_time is None:
            raise Exception("""%s._settle_time must be defined. 
                Should be a float specifying the amount of time (in seconds) 
                to allow the cell to settle back to baseline after the end of a stimulus.""" % self.__class__.__name__)
        return self._settle_time

    @property
    def baseline_regions(self):
        """A list of (start,stop) time pairs where the recordings in this 
        sync_rec can be expected to be in a quiescent state.

        """
        raise Exception("Must be implemented in subclass.")


class BaselineDistributor(Analyzer):
    """Used to find baseline regions in a trace and distribute them on request.
    """
    def __init__(self, rec):
        self._attach(rec)
        self.rec = rec
        self.baselines = rec.baseline_regions
        self.ptr = 0

    def get_baseline_chunk(self, duration=20e-3):
        """Return the (start, stop) indices of a chunk of unused baseline with the
        given duration.
        """
        while True:
            if len(self.baselines) == 0:
                return None
            start, stop = self.baselines[0]
            chunk_start = max(start, self.ptr)
            chunk_stop = chunk_start + duration
            if chunk_stop <= stop:
                self.ptr = chunk_stop
                return chunk_start, chunk_stop
            else:
                self.baselines.pop(0)