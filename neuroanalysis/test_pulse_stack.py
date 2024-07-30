from __future__ import annotations

import h5py
import numpy as np

from neuroanalysis.test_pulse import PatchClampTestPulse


class H5BackedTestPulseStack:
    """A caching, HDF5-backed stack of test pulses."""

    def __init__(self, h5_group: h5py.Group):
        self._h5_group = h5_group
        self._test_pulses: dict[float, PatchClampTestPulse | None] = {}
        # pre-cache just the names of existing test pulses from the file
        for fh in h5_group:
            self._test_pulses[float(fh)] = None

    def __getitem__(self, key: float) -> PatchClampTestPulse:
        if key not in self._test_pulses:
            raise KeyError(f"Test pulse at time {key} not found")
        if self._test_pulses[key] is None:
            # load the test pulse from the file
            tp = self._h5_group[str(key)]
            rec = dict(tp.attrs.items())
            rec['time_values'] = tp[:, 0]
            rec['data'] = tp[:, 1]
            rec['stimulus'] = {k[9:]: v for k, v in tp.attrs.items() if k.startswith('stimulus_')}
            self._test_pulses[key] = PatchClampTestPulse.load(rec)
        return self._test_pulses[key]

    def __len__(self) -> int:
        return len(self._test_pulses)

    def close(self):
        self._h5_group.file.close()

    def flush(self):
        self._h5_group.file.flush()

    def append(self, test_pulse: PatchClampTestPulse) -> None:
        """Append a test pulse to the stack."""
        rec = test_pulse.dump()
        data = np.column_stack((rec['time_values'], rec['data']))
        tp = self._h5_group.create_dataset(
            str(rec['start_time']),
            data=data,
            compression='gzip',
            compression_opts=9,
        )
        for k, v in rec.items():
            if k not in ('time_values', 'data', 'stimulus'):
                tp.attrs[k] = v or 0  # None values are not allowed
        for k, v in rec['stimulus'].items():
            tp.attrs[f"stimulus_{k}"] = v

        # Update the cache
        self._test_pulses[test_pulse.start_time] = test_pulse

    def at_time(self, when: float) -> PatchClampTestPulse | None:
        """Return the test pulse at or immediately previous to the provided time."""
        keys = np.array(list(self._test_pulses.keys()))  # Todo cache this?
        idx = np.searchsorted(keys, when)
        print(f"at_time({when}) idx: {idx}")
        if idx == 0:
            return None
        if idx == len(keys):
            idx -= 1
        return self[keys[idx - 1]]
