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
        self.baselines = [list(r) for r in rec.baseline_regions]

    def get_baseline_chunk(self, duration=20e-3):
        """Return the (start, stop) indices of a chunk of unused baseline with the
        given duration.
        """
        for i, baseline_rgn in enumerate(self.baselines):
            rgn_start, rgn_stop = baseline_rgn
            if rgn_stop - rgn_start < duration:
                continue
            chunk_stop = rgn_start + duration
            baseline_rgn[0] = chunk_stop
            return rgn_start, chunk_stop
        
        # coundn't find any baseline data of the requested length
        return None

    def baseline_chunks(self, duration=20e-3):
        """Iterator yielding (start, stop) indices of baseline chunks.
        """
        while True:
            chunk = self.get_baseline_chunk(duration)
            if chunk is None:
                break
            yield chunk
