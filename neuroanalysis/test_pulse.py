import numpy as np
import scipy.optimize

from .data import PatchClampRecording, TSeries
from .fitting.exp import exp_fit, exp_decay
from .stimuli import find_square_pulses


class PatchClampTestPulse(PatchClampRecording):
    """A PatchClampRecording that contains a subthreshold, square pulse stimulus.
    """
    def __init__(self, rec: PatchClampRecording, indices=None):
        self._parent_recording = rec
        
        if indices is None:
            indices = (0, len(rec['primary']))
        self._indices = indices
        start, stop = indices
        
        pri = rec['primary'][start:stop]
        cmd = rec['command'][start:stop]
        
        # find pulse
        pulses = find_square_pulses(cmd)        
        if len(pulses) == 0:
            raise ValueError("Could not find square pulse in command waveform.")
        elif len(pulses) > 1:
            raise ValueError("Found multiple square pulse in command waveform.")
        pulse = pulses[0]
        pulse.description = 'test pulse'
        
        PatchClampRecording.__init__(self,
            device_type=rec.device_type, 
            device_id=rec.device_id,
            start_time=rec.start_time,
            channels={'primary': pri, 'command': cmd}
        )        
        self._meta['stimulus'] = pulse
                           
        for k in ['clamp_mode', 'holding_potential', 'holding_current', 'bridge_balance',
                  'lpf_cutoff', 'pipette_offset']:
            self._meta[k] = rec._meta[k]
            
        self._analysis = None
        
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
 
    @property
    def parent(self):
        """The recording in which this test pulse is embedded.
        """
        return self._parent_recording

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
        self.pulse_tseries = pulse
        
        # Exponential fit

        # predictions
        base_median = np.median(base.data)
        # access_r = 10e6
        # input_r = 200e6
        # if clamp_mode == 'vc':
        #     # ari = pulse_amp / access_r
        #     # iri = pulse_amp / input_r
        #     # params = {
        #     #     'xoffset': (pulse.t0, 'fixed'),
        #     #     'yoffset': base_median + iri,
        #     #     'amp': ari - iri,
        #     #     'tau': (1e-3, 0.1e-3, 50e-3),
        #     # }
        #     pass
        # else:  # current clamp
        #     # arv = pulse_amp * (access_r - bridge)
        #     # irv = pulse_amp * input_r
        #     # params = {
        #     #     'xoffset': pulse.t0,
        #     #     'yoffset': base_median+arv+irv,
        #     #     'amp': -irv,
        #     #     'tau': (10e-3, 1e-3, 50e-3),
        #     # }
            
        # fit_kws = {'tol': 1e-4}
        
        # ignore initial transients when fitting
        fit_region = pulse.time_slice(pulse.t0 + 150e-6, None)

        # fit the exponential decay
        result = exp_fit(fit_region)
        self.fit_result = result
        # exp curve using fit parameters
        self.fit_trace = TSeries(
            result['model'](fit_region.time_values), 
            time_values=fit_region.time_values)
        # model the exp curve using initial parameters
        self.initial_fit_trace = TSeries(
            exp_decay(fit_region.time_values, *result['initial_guess']),
            time_values=fit_region.time_values)

        # final fit parameters
        fit_yoffset, fit_amp, fit_tau = result['fit']

        # custom double-exp fit to capture pipette transients
        prepulse_median = np.median(data.time_slice(pulse_start-5e-3, pulse_start).data)
        def dexp_decay(t, yoffset, tau, xoffset):
            amp = prepulse_median - yoffset
            return exp_decay(t, yoffset, amp, tau, xoffset) + result['model'](t) - yoffset
        y0 = self.fit_result['model'](pulse.t0)
        guess_amp = base_median - y0
        initial_guess = (
            y0, 
            10e-6,
            pulse_start,
        )
        bounds = (
            [base_median, 0, pulse_start-5e-6], 
            [y0-guess_amp, 200e-6, pulse_start+100e-6]
        )
        for i in range(len(bounds[0])):
            bounds[0][i], bounds[1][i] = min(bounds[0][i], bounds[1][i]), max(bounds[0][i], bounds[1][i])
        pulse_pip_transient = data.time_slice(pulse_start, pulse_start + 5e-3)

        fit = scipy.optimize.curve_fit(
            f=dexp_decay,
            xdata=pulse_pip_transient.time_values, 
            ydata=pulse_pip_transient.data, 
            p0=initial_guess, 
            bounds=bounds, 
            # ftol=1e-8, gtol=1e-8,
        )

        transient_start = fit[0][2]
        tvals = np.arange(transient_start, pulse_stop-padding, dt)
        self.fit_trace = TSeries(dexp_decay(tvals, *fit[0]), time_values=tvals)
        self.initial_fit_trace = TSeries(dexp_decay(tvals, *initial_guess), time_values=tvals)
        pip_transient_yoffset, pip_transient_tau, pip_transient_xoffset = fit[0]

        ### fit again using shorter data
        ### this should help to avoid fitting against h-currents
        #tau4 = fit1[0][2]*10
        #t0 = pulse.xvals('Time')[0]
        #shortPulse = pulse['Time': t0:t0+tau4]
        #if shortPulse.shape[0] > 10:  ## but only if we can get enough samples from this
            #tVals2 = shortPulse.xvals('Time')-params['delayTime']
            #fit1 = scipy.optimize.leastsq(
                #lambda v, t, y: y - expFn(v, t), pred1, 
                #args=(tVals2, shortPulse['primary'].view(np.ndarray) - baseMean),
                #maxfev=200, full_output=1)

        ## Handle analysis differently depending on clamp mode
        if clamp_mode == 'vc':
            hp = self.meta['holding_potential']
            if hp is not None:
                # we can only report base voltage if metadata includes holding potential
                base_v = self['command'].data[0] + hp
            else:
                base_v = None
            base_i = base_median
            
            input_step = fit_yoffset - base_i
            
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
            cap = fit_tau * (1 / access_r + 1 / input_r)

        else:  # IC mode
            base_v = base_median
            hc = self.meta['holding_current']
            if hc is not None:
                # we can only report base current if metadata includes holding current
                base_i = self['command'].data[0] + hc
            else:
                base_i = None
            # y0 = self.fit_result['model'](pulse_start)
            y0 = pip_transient_yoffset
            
            if pulse_amp >= 0:
                v_step = max(1e-5, fit_yoffset - y0)
            else:
                v_step = min(-1e-5, fit_yoffset - y0)
                
            if pulse_amp == 0:
                pulse_amp = 1e-14
                
            input_r = (v_step / pulse_amp)
            access_r = ((y0 - prepulse_median) / pulse_amp) + self.meta['bridge_balance']
            cap = fit_tau / input_r

        self._analysis = {
            'steady_state_resistance': input_r + access_r,
            'input_resistance': input_r,
            'access_resistance': access_r,
            'capacitance': cap,
            'time_constant': fit_tau,
            'fit_yoffset': fit_yoffset,
            'fit_xoffset': pulse.t0,
            'fit_amplitude': fit_amp,
            'baseline_potential': base_v,
            'baseline_current': base_i,
        }
        self._fit_result = result
    
    def plot(self):
        self.analysis
        import pyqtgraph as pg
        name, units = ('pipette potential', 'V') if self.clamp_mode == 'ic' else ('pipette current', 'A')
        plt = pg.plot(labels={'left': (name, units), 'bottom': ('time', 's')})
        plt.plot(self['primary'].time_values, self['primary'].data)
        plt.plot(self.fit_trace.time_values, self.fit_trace.data, pen='b')
