from __future__ import annotations

import os

import h5py
import numpy as np

from neuroanalysis.test_pulse import PatchClampTestPulse


class H5BackedTestPulseStack:
    """A caching, HDF5-backed stack of test pulses."""

    def __init__(self, h5_group: h5py.Group):
        self._containing_groups = [h5_group]
        self._test_pulses: dict[float, PatchClampTestPulse | None] = {}
        # pre-cache just the names of existing test pulses from the file
        for fh in h5_group:
            self._test_pulses[float(fh)] = None

    def __getitem__(self, key: float) -> PatchClampTestPulse:
        if key not in self._test_pulses:
            raise KeyError(f"Test pulse at time {key} not found")
        if self._test_pulses[key] is None:
            # load the test pulse from the file
            tp = next(grp[str(key)] for grp in self._containing_groups if str(key) in grp)
            rec = dict(tp.attrs.items())
            rec['time_values'] = tp[:, 0]
            rec['data'] = tp[:, 1]
            rec['stimulus'] = {k[9:]: v for k, v in tp.attrs.items() if k.startswith('stimulus_')}
            self._test_pulses[key] = PatchClampTestPulse.load(rec)
        return self._test_pulses[key]

    def merge(self, other: H5BackedTestPulseStack):
        """Merge another stack into this one."""
        self._containing_groups += other._containing_groups
        # sort the groups by time to make append logic easy
        self._containing_groups.sort(key=lambda grp: os.path.getmtime(grp.file.filename))
        self._test_pulses.update(other._test_pulses)

    def __len__(self) -> int:
        return len(self._test_pulses)

    def close(self):
        for grp in self._containing_groups:
            grp.file.close()

    def flush(self):
        for grp in self._containing_groups:
            grp.file.flush()

    def append(self, test_pulse: PatchClampTestPulse) -> str:
        """Append a test pulse to the stack. Returns the full path name of the dataset."""
        rec = test_pulse.dump()
        data = np.column_stack((rec['time_values'], rec['data']))
        dataset = self._containing_groups[-1].create_dataset(
            str(rec['start_time']),
            data=data,
            compression='gzip',
            compression_opts=9,
        )
        for k, v in rec.items():
            if k not in ('time_values', 'data', 'stimulus'):
                dataset.attrs[k] = v or 0  # None values are not allowed
        for k, v in rec['stimulus'].items():
            dataset.attrs[f"stimulus_{k}"] = v

        self._test_pulses[test_pulse.start_time] = test_pulse

        return f"{dataset.file.filename}:{dataset.name}"

    def at_time(self, when: float) -> PatchClampTestPulse | None:
        """Return the test pulse at or immediately previous to the provided time."""
        keys = np.array(list(self._test_pulses.keys()))  # Todo cache this?
        idx = np.searchsorted(keys, when)
        if idx == 0:
            return None
        if idx == len(keys):
            idx -= 1
        return self[keys[idx - 1]]
