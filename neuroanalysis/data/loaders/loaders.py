import h5py
import numpy as np
from collections import OrderedDict
from neuroanalysis.data.dataset import SyncRecording, PatchClampRecording, Recording, TSeries
import neuroanalysis.util.mies_nwb_parsing as parser
#import aisynphys.pipeline.opto.data_model as dm
import neuroanalysis.util.device_config as dm
import neuroanalysis.stimuli as stimuli
from neuroanalysis.test_pulse import PatchClampTestPulse

class DatasetLoader():
    """An abstract base class for Dataset loaders."""

    def get_sync_recordings(self, dataset):
        """Return a list of SyncRecordings."""
        raise NotImplementedError("Must be implemented in subclass.")

    def get_recordings(self, sync_rec):
        """Return a dict of {device: Recording}"""
        raise NotImplementedError("Must be implemented in subclass.")

    def get_tseries_data(self, tseries):
        """Return a numpy array of the data in the tseries."""
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


