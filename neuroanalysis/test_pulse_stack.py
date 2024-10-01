from __future__ import annotations

import json
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
        self._np_timestamp_cache = np.array(list(self._test_pulses.keys()))

    def __getitem__(self, key: float) -> PatchClampTestPulse:
        if key not in self._test_pulses:
            raise KeyError(f"Test pulse at time {key} not found")
        if self._test_pulses[key] is None:
            tp = next(grp[str(key)] for grp in self._containing_groups if str(key) in grp)
            if tp.attrs.get('schema version', (0,))[0] == 2:
                self._test_pulses[key] = self._load_test_pulse_v2(tp)
            else:
                self._test_pulses[key] = self._load_test_pulse_unversioned(tp)
        return self._test_pulses[key]

    def _load_test_pulse_unversioned(self, tp):
        rec = dict(tp.attrs.items())
        rec['time_values'] = tp[:, 0]
        rec['data'] = tp[:, 1]
        rec['stimulus'] = {k[9:]: v for k, v in tp.attrs.items() if k.startswith('stimulus_')}
        return PatchClampTestPulse.load(rec)

    def _load_test_pulse_v2(self, tp):
        state = json.loads(tp.attrs['save'])
        if tp.ndim == 1:
            state['recording']['channels']['primary']['time_values'] = None
            state['recording']['channels']['primary']['data'] = tp[:]
        else:
            state['recording']['channels']['primary']['time_values'] = tp[:, 0]
            state['recording']['channels']['primary']['data'] = tp[:, 1]
        return PatchClampTestPulse.load(state)

    def merge(self, other: H5BackedTestPulseStack):
        """Merge another stack into this one."""
        self._containing_groups += other._containing_groups
        # sort the groups by time to make append logic easy
        self._containing_groups.sort(key=lambda grp: os.path.getmtime(grp.file.filename))
        self._test_pulses.update(other._test_pulses)
        self._np_timestamp_cache = np.concatenate((self._np_timestamp_cache, other._np_timestamp_cache))

    def __len__(self) -> int:
        return len(self._test_pulses)

    def close(self):
        for grp in self._containing_groups:
            grp.file.close()

    @property
    def files(self) -> list[h5py.File]:
        return {grp.file for grp in self._containing_groups}

    def flush(self):
        for grp in self._containing_groups:
            grp.file.flush()

    def append(self, test_pulse: PatchClampTestPulse, retain_data=False) -> tuple[str, str]:
        """Append a test pulse to the stack. Returns the full path name of the dataset."""
        tp_dump = test_pulse.save()
        rec = tp_dump['recording']
        pri = rec['channels']['primary']
        del rec['channels']['command']
        # TODO handle both dt and time_values without clobbering each other
        if pri.get('time_values') is not None:
            data = np.column_stack((pri['time_values'], pri['data']))
        else:
            data = pri['data']
        del rec['channels']['primary']['time_values']
        del rec['channels']['primary']['data']
        dataset = self._containing_groups[-1].create_dataset(
            str(rec['start_time']),
            data=data,
            compression='gzip',
            compression_opts=9,
        )
        dataset.attrs['save'] = json.dumps(tp_dump)
        dataset.attrs['schema version'] = tp_dump['schema version']

        self._test_pulses[test_pulse.recording.start_time] = test_pulse if retain_data else None
        self._np_timestamp_cache = np.append(self._np_timestamp_cache, test_pulse.recording.start_time)

        return dataset.file.filename, dataset.name

    def at_time(self, when: float) -> PatchClampTestPulse | None:
        """Return the test pulse at or immediately previous to the provided time."""
        if not self._readable:
            raise ValueError("This stack is not readable")
        keys = self._np_timestamp_cache
        idx = np.searchsorted(keys, when)
        if idx == 0:
            return None
        if idx == len(keys):
            idx -= 1
        return self[keys[idx - 1]]
