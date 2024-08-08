import functools

import numpy as np

import pyqtgraph as pg

from .data import PatchClampRecording, TSeries
from .fitting.exp import exp_fit, fit_double_exp_decay, fit_with_explicit_hessian, exp_decay, double_exp_fit, \
    exact_fit_exp
from .stimuli import find_square_pulses, SquarePulse


class PatchClampTestPulse(PatchClampRecording):
    """A PatchClampRecording that contains a subthreshold, square pulse stimulus.
    """
    def __init__(self, rec: PatchClampRecording, indices=None, stimulus=None):
        if indices is None:
            indices = (0, len(rec['primary']))
        self._indices = indices
        start, stop = indices

        pri = rec['primary'][start:stop]
        channels = {'primary': pri}
        if stimulus is None:
            cmd = rec['command'][start:stop]
            channels['command'] = cmd
            # find pulse
            pulses = find_square_pulses(cmd)
            if len(pulses) == 0:
                raise ValueError("Could not find square pulse in command waveform. Consider using `indices`.")
            elif len(pulses) > 1:
                raise ValueError("Found multiple square pulse in command waveform. Consider using `indices`.")
            pulse = pulses[0]
            pulse.description = 'test pulse'
            stimulus = pulse

        super().__init__(
            recording=rec,
            device_type=rec.device_type,
            device_id=rec.device_id,
            start_time=rec.start_time,
            channels=channels,
            stimulus=stimulus,
            clamp_mode=rec.clamp_mode,
            holding_potential=rec.meta['holding_potential'],
            holding_current=rec.meta['holding_current'],
            bridge_balance=rec.meta['bridge_balance'],
            lpf_cutoff=rec.meta['lpf_cutoff'],
            pipette_offset=rec.meta['pipette_offset'],
        )
        self._analysis = None
        # expose these for display and debugging
        self._main_fit_region = None
        self.main_fit_result = None
        self.main_fit_trace = None
        self.fit_result_with_transient = None
        self.fit_trace_with_transient = None
        self.initial_double_fit_trace = None

    def dump(self):
        """Return a dictionary with all data needed to reconstruct this object.
        """
        return {
            'device_type': self.device_type,
            'device_id': self.device_id,
            'start_time': self.start_time,
            'stimulus': self.stimulus.dump(),
            'data': self['primary'].data,
            'time_values': self['primary'].time_values,
            'clamp_mode': self.clamp_mode,
            'holding_potential':  self.holding_potential,
            'holding_current':  self.holding_current,
            'bridge_balance':  self._meta['bridge_balance'],
            'lpf_cutoff':  self._meta['lpf_cutoff'],
            'pipette_offset':  self._meta['pipette_offset'],
        }

    @classmethod
    def load(cls, data):
        """Reconstruct a PatchClampTestPulse from data returned by `dump()`.
        """
        stim = SquarePulse(**data['stimulus'])
        rec = PatchClampRecording(
            device_type=data['device_type'],
            device_id=data['device_id'],
            start_time=data['start_time'],
            channels={'primary': TSeries(data['data'], time_values=data['time_values'])},
            stimulus=stim,
            clamp_mode=data['clamp_mode'],
            holding_potential=data['holding_potential'],
            holding_current=data['holding_current'],
            bridge_balance=data['bridge_balance'],
            lpf_cutoff=data['lpf_cutoff'],
            pipette_offset=data['pipette_offset'],
        )
        return cls(rec, stimulus=stim)

    @property
    def indices(self):
        return self._indices

    @property
    def access_resistance(self):
        """The access resistance measured from this test pulse.
        
        Includes the bridge balance resistance if the recording was made in
        current clamp mode.
        """
        return self.analysis['access_resistance']
        
    @property
    def input_resistance(self):
        """The input resistance measured from this test pulse.
        """
        return self.analysis['input_resistance']
    
    @property
    def capacitance(self):
        """The capacitance of the cell measured from this test pulse.
        """
        return self.analysis['capacitance']

    @property
    def time_constant(self):
        """The membrane time constant measured from this test pulse.
        """
        return self.analysis['time_constant']

    @property
    def baseline_potential(self):
        """The potential of the cell membrane measured (or clamped) before
        the onset of the test pulse.
        """
        return self.analysis['baseline_potential']
 
    @property
    def baseline_current(self):
        """The pipette current measured (or clamped) before the onset of the
        test pulse.
        """
        return self.analysis['baseline_current']
 
    @property
    def analysis(self):
        if self._analysis is None:
            self._analyze()
        return self._analysis
 
    def _analyze(self):
        # adapted from ACQ4
        
        pulse_amp = self.stimulus.amplitude
        clamp_mode = self.clamp_mode
        
        data = self['primary']
        
        pulse_start = data.t0 + self.stimulus.start_time
        pulse_stop = pulse_start + self.stimulus.duration

        # Extract specific time segments
        padding = 50e-6
        base = data.time_slice(None, pulse_start-padding)
        pulse = data.time_slice(pulse_start+padding, pulse_stop-padding)
        base_median = np.median(base.data)
        prepulse_median = np.median(data.time_slice(pulse_start-5e-3, pulse_start).data)

        main_fit_amp, main_fit_tau, main_fit_yoffset, initial_transient_curve_y = self.two_pass_exp_fit(
            base_median, data, pulse, pulse_start)

        # Handle analysis differently depending on clamp mode
        if clamp_mode == 'vc':
            base_v = self._meta.get('holding_potential', self['command'].data[0])
            base_i = base_median
            
            input_step = main_fit_yoffset - base_i

            peak_rgn = pulse.time_slice(pulse.t0, pulse.t0 + 1e-3)
            if pulse_amp >= 0:
                input_step = max(1e-16, input_step)
                access_step = peak_rgn.data.max() - base_i
                access_step = max(1e-16, access_step)
            else:
                input_step = min(-1e-16, input_step)
                access_step = peak_rgn.data.min() - base_i
                access_step = min(-1e-16, access_step)

            access_r = pulse_amp / access_step
            input_r = (pulse_amp / input_step) - access_r
            cap = main_fit_tau * (1 / access_r + 1 / input_r)

        else:  # IC mode
            base_v = base_median
            base_i = self._meta.get('holding_current')
            if base_i is None:
                base_i = self['command'].data[0]
            y0 = initial_transient_curve_y
            
            if pulse_amp >= 0:
                v_step = max(1e-5, main_fit_yoffset - y0)
            else:
                v_step = min(-1e-5, main_fit_yoffset - y0)
                
            if pulse_amp == 0:
                pulse_amp = 1e-14
                
            input_r = (v_step / pulse_amp)
            access_r = ((y0 - prepulse_median) / pulse_amp) + self.meta['bridge_balance']
            cap = main_fit_tau / input_r

        self._analysis = {
            'start_time': self.start_time,
            'steady_state_resistance': input_r + access_r,
            'input_resistance': input_r,
            'access_resistance': access_r,
            'capacitance': cap,
            'time_constant': main_fit_tau,
            'fit_yoffset': main_fit_yoffset,
            'fit_xoffset': pulse.t0,
            'fit_amplitude': main_fit_amp,
            'baseline_potential': base_v,
            'baseline_current': base_i,
        }

    def two_pass_exp_fit(self, base_median, data, pulse, pulse_start):
        # start by fitting the exponential decay from the post-pipette capacitance, ignoring initial transients
        main_fit_region = pulse.time_slice(pulse.t0 + 150e-6, None)
        self._main_fit_region = main_fit_region
        # self.main_fit_result = exp_fit(main_fit_region)
        # self.main_fit_result = fit_with_explicit_hessian(main_fit_region)
        self.main_fit_result = exact_fit_exp(main_fit_region)
        main_fit_yoffset, main_fit_amp, main_fit_tau = self.main_fit_result['fit']
        self.main_fit_trace = TSeries(self.main_fit_result['model'](main_fit_region.time_values),
                                      time_values=main_fit_region.time_values)
        # now fit with the initial transients included as an additional exponential decay
        try:
            self.fit_result_with_transient = fit_double_exp_decay(
                data, pulse, base_median, pulse_start, self.main_fit_result['model'])
            initial_transient_curve_y = self.fit_result_with_transient['fit'][0]

            self.fit_trace_with_transient = TSeries(
                self.fit_result_with_transient['model'](pulse.time_values), time_values=pulse.time_values)
            # self.initial_double_fit_trace = TSeries(
            #     np.abs(self.fit_result_with_transient['model'](pulse.time_values))
            #     - np.abs(self.main_fit_result['model'](pulse.time_values))
            #     + base_median,
            #     time_values=pulse.time_values)
            self.initial_double_fit_trace = TSeries(
                self.fit_result_with_transient['guessed_model'](pulse.time_values), time_values=pulse.time_values)
        except ValueError:
            initial_transient_curve_y = self.main_fit_result['model'](pulse.t0)
        return main_fit_amp, main_fit_tau, main_fit_yoffset, initial_transient_curve_y

    def one_pass_exp_fit(self, base_median, data, pulse, pulse_start):
        fit_result = double_exp_fit(pulse, pulse_start)
        self._main_fit_region = pulse
        self.main_fit_result = fit_result
        pip_yoffset, pip_xoffset, pip_tau, cell_offset, cell_scale, cell_tau = self.main_fit_result['fit']
        self.main_fit_trace = TSeries(self.main_fit_result['model'](pulse.time_values), time_values=pulse.time_values)
        return cell_scale, cell_tau, cell_offset, pip_yoffset

    @property
    def plot_units(self):
        return 'A' if self.clamp_mode == 'vc' else 'V'

    @property
    def plot_title(self):
        return 'current' if self.clamp_mode == 'vc' else 'potential'

    def plot(self):
        assert self.analysis is not None
        plt = pg.plot(labels={'left': (self.plot_title, self.plot_units), 'bottom': ('time', 's')})
        plt.addLegend()
        plt.plot(self['primary'].time_values, self['primary'].data, name="raw")
        if self.fit_trace_with_transient is not None:
            plt.plot(self.fit_trace_with_transient.time_values, self.fit_trace_with_transient.data, pen='b', name="fit w/ trans")
            plt.plot(self.initial_double_fit_trace.time_values, self.initial_double_fit_trace.data, pen='g', name="initial double fit")
        plt.plot(self.main_fit_trace.time_values, self.main_fit_trace.data, pen='r', name="first fit")
        return plt
