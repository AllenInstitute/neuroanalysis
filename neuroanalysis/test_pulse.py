import contextlib
import numpy as np
import warnings

import pyqtgraph as pg
from .data import PatchClampRecording, TSeries
from .fitting.exp import double_exp_fit, \
    exact_fit_exp
from .stimuli import find_square_pulses, SquarePulse


class LowConfidenceFitError(Exception):
    pass


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

        try:
            main_fit_amp, main_fit_tau, main_fit_yoffset, fit_y0 = self.two_pass_exp_fit(
                base_median, data, pulse, pulse_start)
        except LowConfidenceFitError:
            main_fit_amp, main_fit_tau, main_fit_yoffset, fit_y0 = self.bath_fit(base_median, pulse)

        # Handle analysis differently depending on clamp mode
        if clamp_mode == 'vc':
            base_v = self._meta.get('holding_potential')
            if base_v is None:
                base_v = self['command'].data[0]
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

            access_r = pulse_amp / access_step  # pipette
            input_r = (pulse_amp / input_step) - access_r  # soma
            # This uses the full formula for a parallel RC circuit as derived by
            # https://www.youtube.com/watch?v=4I5hswA45CM
            cap = main_fit_tau * (1 / access_r + 1 / input_r)

        else:  # IC mode
            base_v = base_median
            base_i = self._meta.get('holding_current')
            if base_i is None:
                base_i = self['command'].data[0]

            if pulse_amp >= 0:
                v_step = max(1e-5, main_fit_yoffset - fit_y0)
            else:
                v_step = min(-1e-5, main_fit_yoffset - fit_y0)
                
            if pulse_amp == 0:
                pulse_amp = 1e-14
                
            input_r = v_step / pulse_amp  # soma
            access_r = ((fit_y0 - prepulse_median) / pulse_amp) + self.meta['bridge_balance']  # pipette
            # This uses the formula for a series RC circuit, effectively ignoring the access resistance and even the
            # voltage source. See https://www.youtube.com/watch?v=2m1emG-agbM for derivation.
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

    def _analysis_labels(self):
        return {
            'steady_state_resistance': ('Ω', 'Rss'),
            'input_resistance': ('Ω', 'Ri'),
            'access_resistance': ('Ω', 'Ra'),
            'capacitance': ('F', 'Cm'),
            'time_constant': ('s', 'τ'),
            'fit_yoffset': (self.plot_units, 'Yo'),
            'fit_xoffset': ('s', 'Xo'),
            'fit_amplitude': ('', 'Ya'),
            'baseline_potential': ('V', 'Vh'),
            'baseline_current': ('A', 'Ih'),
        }

    def two_pass_exp_fit(self, base_median, data, pulse, pulse_start):
        # start by fitting the exponential decay from the post-pipette capacitance, ignoring initial transients
        main_fit_region = pulse.time_slice(pulse.t0 + 150e-6, None)
        self._main_fit_region = main_fit_region
        with warnings.catch_warnings(action='ignore'):
            self.main_fit_result = exact_fit_exp(main_fit_region)
        main_fit_yoffset, main_fit_amp, main_fit_tau = self.main_fit_result['fit']
        self.main_fit_trace = TSeries(self.main_fit_result['model'](main_fit_region.time_values),
                                      time_values=main_fit_region.time_values)
        y0 = self.main_fit_result['model'](pulse.t0)
        # TODO doing anything with this transient fit doesn't help pass any tests, and in fact causes a
        #  bunch of failures. not returning any of this for now, but it does plot well.
        with contextlib.suppress(ValueError):
            # now fit with the access transients included as an additional exponential decay
            prediction = self.main_fit_result['model'](pulse.time_values)
            self.fit_result_with_transient = exact_fit_exp(pulse - prediction)

            self.fit_trace_with_transient = TSeries(
                self.fit_result_with_transient['model'](pulse.time_values) + prediction,
                time_values=pulse.time_values,
            )
        if self.main_fit_result['confidence'] < 0.15:
            raise LowConfidenceFitError(self.main_fit_result['confidence'])
        return main_fit_amp, main_fit_tau, main_fit_yoffset, y0

    def bath_fit(self, base_median, pulse):
        # no cell, no non-transient exponential decay, just ohm's law.
        start_y = base_median
        end_y = pulse.data[-len(pulse.data) // 100:].mean()
        yscale = start_y - end_y
        yoffset = end_y
        tau = float('nan')
        y0 = base_median
        return yscale, tau, yoffset, y0

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

    def plot(self, plt=None, label=True):
        assert self.analysis is not None
        if plt is None:
            plt = pg.plot(labels={'left': (self.plot_title, self.plot_units), 'bottom': ('time', 's')})
            plt.addLegend()
        plt.plot(self['primary'].time_values, self['primary'].data, name="raw")
        if self.fit_trace_with_transient is not None:
            plt.plot(self.fit_trace_with_transient.time_values, self.fit_trace_with_transient.data, pen='b', name="fit w/ trans")
        if self.initial_double_fit_trace is not None:
            plt.plot(self.initial_double_fit_trace.time_values, self.initial_double_fit_trace.data, pen='g', name="initial double fit")
        plt.plot(self.main_fit_trace.time_values, self.main_fit_trace.data, pen='r', name="first fit")
        if label:
            self.label_for_plot(plt.getPlotItem())
        return plt

    def label_for_plot(self, plt):
        asymptote = self.analysis['fit_yoffset']
        plt.addItem(pg.InfiniteLine(
            (0, asymptote),
            angle=0,
            pen=pg.mkPen((180, 180, 240), dash=[3, 4]),
        ))
        abbrevs = self._analysis_labels()
        text = "Estimated:<br/>" + "<br/>".join([
            f"{abbrevs[key][1]}: {pg.siFormat(val, suffix=abbrevs[key][0])}"
            for key, val in self.analysis.items()
            if val is not None and key in abbrevs
        ])
        label = pg.LabelItem(text.strip(), color=(180, 180, 240))
        label.setParentItem(plt.vb)
        label.setPos(5, 5)
        return label
