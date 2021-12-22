import numpy as np
from neuroanalysis.analyzers.analyzer import Analyzer
from neuroanalysis.stimuli import find_square_pulses, find_noisy_square_pulses, SquarePulse
from neuroanalysis.spike_detection import detect_evoked_spikes


class GenericStimPulseAnalyzer(Analyzer):
    """For analyzing noise-free or noisy square-pulse stimulations."""

    def __init__(self, rec):
        self._attach(rec)
        self.rec = rec
        self._pulses = {}

    def _check_channel(self, channel):
        if channel not in self.rec.channels:
            if channel is None:
                raise ValueError("Please specify which channel to analyze. Options are: %s" % self.rec.channels)
            else:
                raise ValueError("Recording %s does not contain specified channel (%s). Options are: %s" % (self.rec, channel, self.rec.channels))

    def pulses(self, channel=None):
        """Return a list of (start_time, stop_time, amp) tuples describing square pulses
        in the specified channel.
        """
        self._check_channel(channel)

        if self._pulses.get(channel) is None:
            trace = self.rec[channel]
            if trace.data[:10].std() > 0:
                pulses = find_noisy_square_pulses(trace, std_threshold=10)
            else:
                pulses = find_square_pulses(trace)
            self._pulses[channel] = []
            for p in pulses:
                start = p.global_start_time
                stop = p.global_start_time + p.duration
                self._pulses[channel].append((start, stop, p.amplitude))
        return self._pulses[channel]


class PWMStimPulseAnalyzer(GenericStimPulseAnalyzer):
    """For analyzing noise-free digital channels where pulse width modulation may have
     been used to modulate amplitude."""

    def __init__(self, rec, pwm_min_frequency=1000.):
        GenericStimPulseAnalyzer.__init__(self, rec)
        self._pwm_params = {}

        #### change the pwm_min_frequency to change what is considered pulse-width-modulation
        ##      - pulses with frequencies equal to or greater than self.pwm_min_frequency 
        ##        will be considered pulse-width-modulation, and will be grouped into stimulation pulses with amplitudes ranging from 0-1
        self.pwm_min_frequency = pwm_min_frequency #Hz


        self.pwm_min_delay = 1./self.pwm_min_frequency + 1e-6 ##(+ a buffer in case of floating point error)

    def pulses(self, channel=None):
        """Return a list of SquarePulses found in the given channel of the recording. 
        If there is pulse-width modulation (higher than %s Hz), it will be grouped
        into a single SquarePulse with lower amplitude.
            Example:
                _____|||||_______________|||||_____________|||||____________
                     <--->               <--->             <--->

                The trace shown above has pulse width modulation would return a 
                list of 3 SquarePulses with an amplitude equal to the mean of the 
                trace during the periods indicated by the arrows.

                The trace shown below (no pulse width modulation) would also 
                return a list of 3 SquarePulses, but with an amplitude of 1. 

                     _____               _____             _____
                ____|     |_____________|     |___________|     |___________

            Parameters:
            -----------

            channel : str | None
                The channel to analyze pulses from. 
            """ % str(self.pwm_min_frequency)

        self._check_channel(channel)

        if self._pulses.get(channel) is None:
            trace = self.rec[channel]

            if trace.data[:10].std() > 0:
                all_pulses = find_noisy_square_pulses(trace)
            else:
                all_pulses = find_square_pulses(trace)

            ## figure out if there is pwm happening
            pwm = False
            if len(all_pulses) > 1:
                for i in range(len(all_pulses)-1):
                    if all_pulses[i+1].global_start_time - all_pulses[i].global_start_time <= self.pwm_min_delay: ## dealing with pwm
                        pwm = True
                        break

            ## convert pwm pulses into single stimulation pulses
            if pwm:
                pulses = []
                self._pwm_params[channel] = []
                ### make an array of start times
                starts = np.array([p.global_start_time for p in all_pulses])

                ### do a diff on the array - look for points larger than 1/min_frequency
                breaks = np.argwhere(np.diff(starts) > self.pwm_min_delay) ## gives indices of the last pulse in a pwm pulse
                if len(breaks) == 0: ## only one pulse
                    pulse, params = self._create_pulse_from_pwm(all_pulses)
                    pulses.append(pulse)
                    self._pwm_params[channel].append(params)


                ### take the pulses between large diffs and turn them into one pulse with appropriate duration and amplitude
                else:
                    start_i = 0
                    for i, b in enumerate(breaks):
                        pulse, params = self._create_pulse_from_pwm(all_pulses[start_i:b+1])
                        pulses.append(pulse)
                        self._pwm_params[channel].append(params)
                        start_i = b+1

                    pulse, params = self._create_pulse_from_pwm(all_pulses[start_i:])
                    pulses.append(pulse)
                    self._pwm_params[channel].append(params)

            else:
                self._pwm_params[channel] = None
                pulses = [SquarePulse(start_time=p.global_start_time, duration=p.duration, amplitude=1, units='percent') for p in all_pulses]

            ### convert from SquarePulse to (start, stop, amplitude)
            self._pulses[channel] = pulses

        return self._pulses[channel]

    def _create_pulse_from_pwm(self, pwms):
        """Return a (SquarePulse, pwm_param_dict) where pwm_param_dict has 'frequency' and 'duration' for pwm pulses."""
        dt = pwms[1].global_start_time - pwms[0].global_start_time
        duration = dt*len(pwms)
        amplitude = pwms[0].duration / dt
        return (SquarePulse(start_time=pwms[0].global_start_time, duration=duration, amplitude=amplitude, units='percent'),
                {'frequency': 1. / dt,
                 'duration':pwms[0].duration
                })


    def pwm_params(self, channel=None, pulse_n=None):
        """Return frequency and duration of pulse width modulation pulses for the given channel and pulse number.
        """
        self._check_channel(channel)

        if self._pulses.get(channel) is None: ## we've not analyzed this channel yet, do it now
            self.pulses(channel=channel)

        params = self._pwm_params[channel]
        if params is None:
            return None

        if pulse_n is None:
            params = set(params)
            if len(params) == 1:
                return params
            else:
                raise Exception("Please specify pulse number to return params for. Found %i different param sets in %s channel"%(len(params), channel))
        else:
            return self._pwm_params[channel][pulse_n]



class PatchClampStimPulseAnalyzer(GenericStimPulseAnalyzer):
    """Used for analyzing a patch clamp recording with square-pulse stimuli.
    """
    def __init__(self, rec):
        GenericStimPulseAnalyzer.__init__(self, rec)
        self._evoked_spikes = None
        
    def pulses(self, channel='command'):
        """Return a list of (start_time, stop_time, amp) tuples describing square pulses
        in the stimulus.
        """
        if self._pulses.get(channel) is None:
            trace = self.rec[channel]
            pulses = find_square_pulses(trace)
            self._pulses[channel] = []
            for p in pulses:
                start = p.global_start_time
                stop = p.global_start_time + p.duration
                self._pulses[channel].append((start, stop, p.amplitude))
        return self._pulses[channel]

    def pulse_chunks(self):
        """Return time-slices of this recording where evoked spikes are expected to be found (one chunk
        per pulse)
        
        Each recording returned has extra metadata keys added: 
        - pulse_edges: start/end times of the stimulus pulse
        - pulse_amplitude: amplitude of stimulus puse (in V or A)
        - pulse_n: the number of this pulse (all detected square pulses are numbered in order from 0)

        """
        pre_trace = self.rec['primary']

        # Detect pulse times
        pulses = self.pulses()

        # filter out test pulse if it exists
        stim_pulses = pulses[1:] if self.rec.has_inserted_test_pulse else pulses
        
        # cut out a chunk for each pulse
        chunks = []
        for i,pulse in enumerate(stim_pulses):
            pulse_start_time, pulse_end_time, amp = pulse
            if amp < 0:
                # assume negative pulses do not evoke spikes
                # (todo: should be watching for rebound spikes as well)
                continue
            # cut out a chunk of the recording for spike detection
            start_time = pulse_start_time - 2e-3
            stop_time = pulse_end_time + 4e-3
            if i < len(stim_pulses) - 1:
                # truncate chunk if another pulse is present
                next_pulse_time = stim_pulses[i+1][0]
                stop_time = min(stop_time, next_pulse_time)
            chunk = self.rec.time_slice(start_time, stop_time)
            chunk.meta['pulse_edges'] = [pulse_start_time, pulse_end_time]
            chunk.meta['pulse_amplitude'] = amp
            chunk.meta['pulse_n'] = i
            chunks.append(chunk)
            
        return chunks

    def evoked_spikes(self):
        """Given presynaptic Recording, detect action potentials
        evoked by current injection or unclamped spikes evoked by a voltage pulse.

        Returns
        -------
        spikes : list
            [{'pulse_n', 'pulse_start', 'pulse_end', 'spikes': [...]}, ...]
        """
        if self._evoked_spikes is None:
            spike_info = []
            for i,chunk in enumerate(self.pulse_chunks()):
                pulse_edges = chunk.meta['pulse_edges']
                spikes = detect_evoked_spikes(chunk, pulse_edges)
                spike_info.append({'pulse_n': chunk.meta['pulse_n'], 'pulse_start': pulse_edges[0], 'pulse_end': pulse_edges[1], 'spikes': spikes})
            self._evoked_spikes = spike_info
        return self._evoked_spikes
