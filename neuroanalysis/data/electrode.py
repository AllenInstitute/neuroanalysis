


class Electrode(object):
    """Represents a single patch clamp electrode.

    Parameters
    ----------
    electrode_id : int
        ID that identifies this electrode uniquely amongst all other electrodes used in the same experiment
    start_time : datetime
        Beginning time of electrode use
    stop_time : datetime
        End of time of electrode use
    device_id : int
        ID of the ephys device connected to this electrode
        (same as neuroanalysis.data.Recording.device_id)
    patch_status : str | None
        Status of patch attempt: No seal, Low seal, GOhm seal, Technical failure, or No attempt
    """
    def __init__(self, electrode_id, start_time, stop_time, device_id, patch_status=None):
        self.electrode_id = electrode_id
        self.start_time = start_time
        self.stop_time = stop_time
        self.device_id = device_id
        self.patch_status = patch_status
        self.cell = None
        self._internal_solution = None

    @property
    def internal_solution(self):
        return self._internal_solution
