import numpy as np

from .data import PatchClampRecording, TSeries
from .fitting.exp import exp_fit, fit_double_exp_decay
from .stimuli import find_square_pulses


class PatchClampTestPulse(PatchClampRecording):
    """A PatchClampRecording that contains a subthreshold, square pulse stimulus.
    """
    def __init__(self, rec: PatchClampRecording, indices=None):
        if indices is None:
            indices = (0, len(rec['primary']))
        self._indices = indices
        start, stop = indices

        pri = rec['primary'][start:stop]
        cmd = rec['command'][start:stop]

        # find pulse
        pulses = find_square_pulses(cmd)
        if len(pulses) == 0:
            raise ValueError("Could not find square pulse in command waveform. Consider using `indices`.")
        elif len(pulses) > 1:
            raise ValueError("Found multiple square pulse in command waveform. Consider using `indices`.")
        pulse = pulses[0]
        pulse.description = 'test pulse'
        super().__init__(
            recording=rec,
            device_type=rec.device_type,
            device_id=rec.device_id,
            start_time=rec.start_time,
            channels={'primary': pri, 'command': cmd},
            stimulus=pulse,
            clamp_mode=rec.clamp_mode,
            holding_potential=rec._meta['holding_potential'],
            holding_current=rec._meta['holding_current'],
            bridge_balance=rec._meta['bridge_balance'],
            lpf_cutoff=rec._meta['lpf_cutoff'],
            pipette_offset=rec._meta['pipette_offset'],
        )
        self._analysis = None
        # expose these for display and debugging
        self.main_fit_result = None
        self.main_fit_trace = None
        self.fit_result_with_transient = None
        self.fit_trace_with_transient = None
        self.initial_double_fit_trace = None

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
        dt = data.dt
        
        # Extract specific time segments
        padding = 50e-6
        base = data.time_slice(None, pulse_start-padding)
        pulse = data.time_slice(pulse_start+padding, pulse_stop-padding)
        base_median = np.median(base.data)
        prepulse_median = np.median(data.time_slice(pulse_start-5e-3, pulse_start).data)

        # start by fitting the exponential decay from the post-pipette capacitance, ignoring initial transients
        main_fit_region = pulse.time_slice(pulse.t0 + 150e-6, None)
        self.main_fit_result = exp_fit(main_fit_region)
        main_fit_yoffset, main_fit_amp, main_fit_tau = self.main_fit_result['fit']
        self.main_fit_trace = TSeries(self.main_fit_result['model'](pulse.time_values), time_values=pulse.time_values)

        # now fit with the initial transients included as an additional exponential decay
        try:
            self.fit_result_with_transient = fit_double_exp_decay(
                data, pulse, base_median, pulse_start, self.main_fit_result['model'])
            transient_yoffset = self.fit_result_with_transient['fit'][0]

            self.fit_trace_with_transient = TSeries(
                self.fit_result_with_transient['model'](pulse.time_values), time_values=pulse.time_values)
            self.initial_double_fit_trace = TSeries(
                self.fit_result_with_transient['guessed_model'](pulse.time_values), time_values=pulse.time_values)
        except ValueError:
            transient_yoffset = self.main_fit_result['model'](pulse.t0)

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
            input_r = pulse_amp / input_step
            cap = main_fit_tau * (1 / access_r + 1 / input_r)

        else:  # IC mode
            base_v = base_median
            base_i = self._meta.get('holding_current', self['command'].data[0])
            # y0 = self.fit_result['model'](pulse_start)
            y0 = transient_yoffset
            
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

    @property
    def plot_units(self):
        return 'A' if self.clamp_mode == 'vc' else 'V'

    @property
    def plot_title(self):
        return 'pipette potential' if self.clamp_mode == 'vc' else 'pipette current'

    def plot(self):
        assert self.analysis is not None
        import pyqtgraph as pg
        plt = pg.plot(labels={'left': (self.plot_title, self.plot_units), 'bottom': ('time', 's')})
        plt.plot(self['primary'].time_values, self['primary'].data)
        plt.plot(self.fit_trace_with_transient.time_values, self.fit_trace_with_transient.data, pen='b')
