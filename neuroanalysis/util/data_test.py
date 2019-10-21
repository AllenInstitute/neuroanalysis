import os, pickle, traceback
import numpy as np


class DataTestCase(object):
    """Base class for tests that work by loading input args and expected results
    from a file.
    """
    def __init__(self, test_function):
        self.compare_opts = {'rtol': 0.01}
        self.test_function = test_function
        self._input_args = None
        self._meta = None
        self._expected_result = None
        self._current_result = None
        self._file_path = None
        self._loaded_file_path = None

    def load_file(self, file_path):
        data = pickle.load(open(file_path, 'rb'))
        self._input_args = data['input_args']
        self._expected_result = data['expected_result']
        self._meta = data['meta']
        self._loaded_file_path = file_path

    def save_file(self, file_path=None):
        if file_path is None:
            file_path = self._loaded_file_path

        info = {
            'input_args': self.input_args,
            'expected_result': self.current_result,
            'meta': self.meta,
        }
        tmpfile = file_path + '.tmp'
        pickle.dump(info, open(tmpfile, 'wb'))
        os.rename(tmpfile, file_path)
        print("Updated test file %s" % file_path)

    @property
    def input_args(self):
        return self._input_args

    @property
    def meta(self):
        return self._meta

    @property
    def expected_result(self):
        return self._expected_result

    @property
    def current_result(self):
        return self._current_result

    def run_test(self):
        self._current_result = self.test_function(**self.input_args)
        self.check_result(self._current_result)

    def audit_test(self, ui):
        ui.clear()

        try:
            # display test data/thresholds and current result in second panel
            self._current_result = self.test_function(ui=ui.display2, **self.input_args)
            self.check_result(self._current_result)


        except Exception as exc:
            traceback.print_exc()
            
            # display test data/thresholds and expected result in first panel
            self.test_function(ui=ui.display1, **self.input_args)

            ui.show_results(self.expected_result, self._current_result)

            print("Expected:", self.expected_result)
            print("Current: ", self.current_result)

            ui.user_passfail()

            # user passed result if we got here
            self.save_file()

    def check_result(self, result):
        self.compare_results(self.expected_result, result, **self.compare_opts)

    def compare_results(self, expected, current, **opts):
        """
        Compare *current* result to *expected* result. 
        
        If *result* and *expected* do not match, then raise an exception.
        """
        # Check test structures are the same
        self.compare_types(expected, current)

        if hasattr(current, '__len__'):
            assert len(current) == len(expected)
            
        if isinstance(current, dict):
            for k in current:
                assert k in expected
            for k in expected:
                assert k in current
                self.compare_results(expected[k], current[k], **opts)
        elif isinstance(current, list):
            for i in range(len(current)):
                self.compare_results(expected[i], current[i], **opts)
        elif isinstance(current, np.ndarray):
            assert current.shape == expected.shape
            if current.dtype.fields is None:
                intnan = -9223372036854775808  # happens when np.nan is cast to int
                inans = np.isnan(current) | (current == intnan)
                enans = np.isnan(expected) | (expected == intnan)
                assert np.all(inans == enans)
                mask = ~inans
                assert np.allclose(current[mask], expected[mask], rtol=opts['rtol'])
            else:
                for k in current.dtype.fields.keys():
                    self.compare_results(expected[k], current[k], **opts)
        elif np.isscalar(current):
            assert np.allclose(expected, current, rtol=opts['rtol'])
        else:
            try:
                assert current == expected
            except AssertionError:
                raise
            except Exception:
                raise NotImplementedError("Cannot compare objects of type %s" % type(current))

    def compare_types(self, a, b):
        if type(a) is type(b):
            return
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            return
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            return
        raise TypeError("Types do not match: %s %s" % (type(a), type(b)))

    @property
    def name(self):
        meta = self.meta
        return "%s_%s_%s_%0.3f" % (meta['expt_id'], meta['sweep_id'], meta['device_id'], self.input_args['pulse_edges'][0])
