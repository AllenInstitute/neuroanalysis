import numpy as np

from neuroanalysis.data import PatchClampRecording, SyncRecording
from neuroanalysis.stimuli import Stimulus
from neuroanalysis.test_pulse import PatchClampTestPulse


class DatasetLoader(object):
    """An abstract base class for Dataset loaders."""
    
    def get_dataset_name(self) -> str:
        """Return a string with the name of this dataset."""
        raise NotImplementedError("Must be implemented in subclass.")

    def get_sync_recordings(self, dataset) -> list[SyncRecording]:
        """Return a tuple (list of SyncRecordings, list of RecordingSequences)."""
        raise NotImplementedError("Must be implemented in subclass.")

    def get_recordings(self, sync_rec) -> dict[str, PatchClampRecording]:
        """Return a dict of {device: Recording}"""
        raise NotImplementedError("Must be implemented in subclass.")

    def get_tseries_data(self, tseries) -> np.ndarray:
        """Return a numpy array of the data in the tseries."""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_stimulus(self, recording) -> Stimulus:
        """Return an instance of stimuli.Stimulus"""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_stimulus_items(self, recording) -> list[Stimulus]:
        """Return a list of Stimulus instances. 
        Used with LazyLoadStimulus to parse stimuli when they are needed."""
        raise NotImplementedError("Must be implemented in subclass.")

    def load_test_pulse(self, recording) -> PatchClampTestPulse:
        """Return a PatchClampTestPulse."""
        raise NotImplementedError("Must be implemented in subclass.")

    def find_nearest_test_pulse(self, recording):
        raise NotImplementedError("Must be implemented in subclass.")

    def get_baseline_regions(self, recording):
        raise NotImplementedError("Must be implemented in subclass.")


