import numpy as np

from .data import PatchClampRecording, TSeries
from .fitting import Exp
from .stimuli import find_square_pulses


class PatchClampTestPulse(PatchClampRecording):
    """A PatchClampRecording that contains a subthreshold, square pulse stimulus.
    """
    def __init__(self, rec, indices=None):
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
        
        meta = self.meta
        pulse_amp = self.stimulus.amplitude
        clamp_mode = self.clamp_mode
        
        data = self['primary']
        pulse_start = data.index_at(self.stimulus.start_time)
        pulse_stop = data.index_at(self.stimulus.start_time + self.stimulus.duration)
        dt = data.dt
        
        # Extract specific time segments
        nudge = int(50e-6 / dt)
        base = data[:pulse_start-nudge]
        pulse = data[pulse_start+nudge:pulse_stop-nudge]
        pulse_end = pulse[int(len(pulse)*2./3.):]  # last 1/3 of pulse response 
        end = data[pulse_stop+nudge:]
        
        # Exponential fit

        # predictions
        base_median = np.median(base.data)
        access_r = 10e6
        input_r = 200e6
        if clamp_mode == 'vc':
            ari = pulse_amp / access_r
            iri = pulse_amp / input_r
            params = {
                'xoffset': (pulse.t0, 'fixed'),
                'yoffset': base_median + iri,
                'amp': ari - iri,
                'tau': (1e-3, 0.1e-3, 50e-3),
            }
        else:
            bridge = meta['bridge_balance']
            arv = pulse_amp * (access_r - bridge)
            irv = pulse_amp * input_r
            params = {
                'xoffset': pulse.t0,
                'yoffset': base_median+arv+irv,
                'amp': -irv,
                'tau': (10e-3, 1e-3, 50e-3),
            }
            
        fit_kws = {'tol': 1e-4}
        model = Exp()
        
        # ignore initial transients when fitting
        fit_region = pulse.time_slice(pulse.t0 + 150e-6, None)
        
        result = model.fit(fit_region.data, x=fit_region.time_values, fit_kws=fit_kws, params=params)
        fit = result.best_values
        err = model.nrmse(result)
        
        self.fit_trace = TSeries(result.eval(), time_values=fit_region.time_values)
        
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
            
            input_step = fit['yoffset'] - base_i
            
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
            
            # No capacitance in VC mode yet; the methods
            # we've tried don't work very well.
            tau = None
            cap = None
        
        else:
            base_v = base_median
            hc = self.meta['holding_current']
            if hc is not None:
                # we can only report base current if metadata includes holding current
                base_i = self['command'].data[0] + hc
            else:
                base_i = None
            y0 = result.eval(x=pulse.t0)
            
            if pulse_amp >= 0:
                v_step = max(1e-5, fit['yoffset'] - y0)
            else:
                v_step = min(-1e-5, fit['yoffset'] - y0)
                
            if pulse_amp == 0:
                pulse_amp = 1e-14
                
            input_r = (v_step / pulse_amp)
            access_r = ((y0 - base_median) / pulse_amp) + bridge
            tau = fit['tau']
            cap = tau / input_r

        self._analysis = {
            'input_resistance': input_r,
            'access_resistance': access_r,
            'capacitance': cap,
            'time_constant': tau,
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
