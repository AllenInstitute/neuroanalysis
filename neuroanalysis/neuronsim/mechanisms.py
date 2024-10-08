# -*- coding: utf-8 -*-
"""
Simple neuron simulator for Python.
Also simulates voltage clamp and current clamp with access resistance.

Luke Campagnola 2015
"""

from collections import OrderedDict

import numpy as np
import scipy.interpolate

from .components import Mechanism, Channel
from ..units import MOhm, us, ms, mS, pF, pA, mV, cm


class PatchClamp(Mechanism):
    type = 'PatchClamp'
    
    def __init__(self, mode='ic', ra=2*MOhm, cpip=0.5*pF, **kwds):
        self.ra = ra
        self.cpip = cpip
        self._mode = mode
        self.cmd_queue = []
        self.last_time = 0
        self.holding = {'ic': 0.0*pA, 'vc': -65*mV}
        self.gain = 50e-6  # arbitrary VC gain
        init_state = OrderedDict([('V', -65*mV)])
        Mechanism.__init__(self, init_state, **kwds)

    def queue_command(self, cmd, dt, start=None):
        """Execute a command as soon as possible.
        
        Return the time at which the command will begin.
        """
        assert cmd.ndim == 1 and cmd.shape[0] > 0
        if len(self.cmd_queue) == 0:
            next_start = self.last_time + dt
        else:
            last_start, last_dt, last_cmd = self.cmd_queue[-1]
            next_start = last_start + len(last_cmd) * last_dt

        if start is None:
            start = next_start
        elif start < next_start:
            raise ValueError(f'Cannot start next command before {next_start:f}; asked for {start:f}.')

        self.cmd_queue.append((start, dt, cmd))
        return start
    
    def queue_commands(self, cmds, dt):
        """Queue multiple commands for execution.
        """
        return [self.queue_command(c, dt) for c in cmds]

    @property
    def mode(self):
        return self._mode
        
    def clear_queue(self):
        self.cmd_queue = []
        
    def set_mode(self, mode):
        self._mode = mode
        self.clear_queue()
        
    def set_holding(self, mode, val):
        if mode not in self.holding:
            raise ValueError("Mode must be 'ic' or 'vc'")
        self.holding[mode] = val

    def current(self, state):
        # Compute current through tip of pipette at this timestep
        vm = state[self.section, 'V']
        ve = state[self, 'V']
        return (ve - vm) / self.ra
    
    def derivatives(self, state):
        t = state['t']
        self.last_time = t
        ## Select between VC and CC
        cmd = self.get_cmd(t)
            
        # determine current generated by voltage clamp 
        if self.mode == 'vc':
            ve = state[self, 'V']
            cmd = (cmd-ve) * self.gain
        
        # Compute change in electrode potential
        dve = (cmd - self.current(state)) / self.cpip
        return [dve]

    def get_cmd(self, t):
        """Return command value at time *t*.
        
        Values are interpolated linearly between command points.
        """
        hold = self.holding[self.mode]
    
        while len(self.cmd_queue) > 0:
            (start, dt, data) = self.cmd_queue[0]
            i1 = int(np.floor((t - start) / dt))
            if i1 < -1:
                # before start of next command; return holding
                return hold
            elif i1 == -1:
                # interpolate from holding into start of next command
                v1 = hold
                vt1 = start - dt
                v2 = data[0] + hold
                vt2 = start
                break
            elif i1 >= len(data):
                # this command has expired; remove and try next command
                self.cmd_queue.pop(0)
                continue
            else:
                v1 = data[i1] + hold
                vt1 = start + i1 * dt
                if i1+1 < len(data):
                    # interpolate to next command point
                    v2 = data[i1+1] + hold
                    vt2 = vt1 + dt
                    break
                else:
                    if len(self.cmd_queue) > 1 and vt1 + dt >= self.cmd_queue[1][0]:
                        # interpolate from command to next command array
                        v2 = self.cmd_queue[1][2][0]
                        vt2 = self.cmd_queue[1][0]
                    else:
                        # interpolate from command back to holding
                        v2 = hold
                        vt2 = vt1 + dt
                    break
                
        if len(self.cmd_queue) == 0:
            return hold
        
        s = (t - vt1) / (vt2 - vt1)
        return v1 * (1-s) + v2 * s
        

class Noise(Mechanism):
    """Injects gaussian noise current.
    
    Note: This incurs a large overhead because it forces the integrator to use
    very small timesteps.
    """
    
    type = 'Inoise'
    
    def __init__(self, mean=0, stdev=5*pA, dt=100*us, **kwds):
        init_state = OrderedDict([])
        Mechanism.__init__(self, init_state, **kwds)
        self.mean = mean
        self.stdev = stdev
        self.dt = dt
        self._start_time = None
        self._end_time = None
        self._noise = None
        
    def current(self, state):
        t = state['t']
        if self._start_time is None or t > self._end_time:
            self._generate_noise(t)
        return self._noise(t)
        
    def derivatives(self, state):
        return []
        
    def _generate_noise(self, t):
        n = int(100*ms / self.dt)
        padding = 10*ms
        t = np.arange(n) * self.dt + (t - padding)
        noise = np.random.normal(size=n, loc=self.mean, scale=self.stdev)
        self._noise = scipy.interpolate.interp1d(t, noise)
        self._start_time = t
        self._end_time = t[-1] - padding


class Leak(Channel):
    type = 'Ileak'
    
    def __init__(self, gbar=0.1*mS/cm**2, erev=-55*mV, **kwds):
        Channel.__init__(self, gbar=gbar, init_state={}, **kwds)
        self.erev = erev

    def open_probability(self, state):
        if state.state.ndim == 2:
            # need to return an array of the correct length..
            return np.ones(state.state.shape[1])
        else:
            return 1

    def derivatives(self, state):
        return []


class HHK(Channel):
    """Hodgkin-Huxley K channel.
    """
    type = 'IK'
    
    max_op = 0.55
    
    @classmethod
    def compute_rates(cls):
        cls.rates_vmin = -100
        cls.rates_vstep = 0.1
        vm = np.arange(cls.rates_vmin, cls.rates_vmin+400, cls.rates_vstep)
        cls.rates = np.empty((len(vm), 2))
        cls.rates[:,0] = (0.1 - 0.01*vm) / (np.exp(1.0 - 0.1*vm) - 1.0)
        cls.rates[:,1] = 0.125 * np.exp(-vm / 80.)
        
    def __init__(self, gbar=12*mS/cm**2, **kwds):
        init_state = OrderedDict([('n', 0.3)]) 
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0
        
    @property
    def erev(self):
        return self.section.ek
        
    def open_probability(self, state):
        return state[self, 'n']**4

    def derivatives(self, state):
        # temperature dependence of rate constants
        q10 = 3 ** ((self.sim.temp-6.3) / 10.)
        vm = state[self.section, 'V'] - self.shift
        
        vm = vm + 65e-3   ## gating parameter eqns assume resting is 0mV
        vm *= 1000.   ##  ..and that Vm is in mV
        
        n = state[self, 'n']
        
        # disabled for now -- does not seem to improve speed.
        #an, bn = self.interpolate_rates(self.rates, vm, self.rates_vmin, self.rates_vstep)
        
        an = (0.1 - 0.01*vm) / (np.exp(1.0 - 0.1*vm) - 1.0)
        bn = 0.125 * np.exp(-vm / 80.)
        dn = q10 * (an * (1.0 - n) - bn * n)
        return [dn*1e3]
                 

class HHNa(Channel):
    """Hodgkin-Huxley Na channel.
    """
    type = 'INa'
    
    max_op = 0.2
    
    @classmethod
    def compute_rates(cls):
        cls.rates_vmin = -100
        cls.rates_vstep = 0.1
        vm = np.arange(cls.rates_vmin, cls.rates_vmin+400, cls.rates_vstep)
        cls.rates = np.empty((len(vm), 4))
        cls.rates[:,0] = (2.5-0.1*vm) / (np.exp(2.5-0.1*vm) - 1.0)
        cls.rates[:,1] = 4. * np.exp(-vm / 18.)
        cls.rates[:,2] = 0.07 * np.exp(-vm / 20.)
        cls.rates[:,3] = 1.0 / (np.exp(3.0 - 0.1 * vm) + 1.0)
        
    def __init__(self, gbar=40*mS/cm**2, **kwds):
        init_state = OrderedDict([('m', 0.05), ('h', 0.6)]) 
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.shift = 0
        
    @property
    def erev(self):
        return self.section.ena
        
    def open_probability(self, state):
        return state[self, 'm']**3 * state[self, 'h']

    def derivatives(self, state):
        # temperature dependence of rate constants
        q10 = 3 ** ((self.sim.temp-6.3) / 10.)
        vm = state[self.section, 'V'] - self.shift
        m = state[self, 'm']
        h = state[self, 'h']

        vm = vm + 65e-3   ## gating parameter eqns assume resting is 0mV
        vm *= 1000.   ##  ..and that Vm is in mV
        
        # disabled for now -- does not seem to improve speed.
        #am, bm, ah, bh = self.interpolate_rates(self.rates, vm, self.rates_vmin, self.rates_vstep)
        
        am = (2.5-0.1*vm) / (np.exp(2.5-0.1*vm) - 1.0)
        bm = 4. * np.exp(-vm / 18.)
        dm = q10 * (am * (1.0 - m) - bm * m)
        
        ah = 0.07 * np.exp(-vm / 20.)
        bh = 1.0 / (np.exp(3.0 - 0.1 * vm) + 1.0)
        dh = q10 * (ah * (1.0 - h) - bh * h)

        return [dm*1e3, dh*1e3]
                 

class IH(Channel):
    """Ih from Destexhe 1993
    """
    type = 'IH'
    
    max_op = 0.3
    
    def __init__(self, gbar=30*mS/cm**2, **kwds):
        init_state = OrderedDict([('f', 0), ('s', 0)]) 
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.erev = -43*mV
        self.shift = 0
        
    def open_probability(self, state):
        return state[self, 'f'] * state[self, 's']
    
    def derivatives(self, state):
        vm = state[self.section, 'V'] - self.shift
        f = state[self, 'f']
        s = state[self, 's']
        
        #vm = vm + 65e-3   ## gating parameter eqns assume resting is 0mV
        vm *= 1000.   ##  ..and that Vm is in mV
        Hinf = 1.0 / (1.0 + np.exp((vm + 68.9) / 6.5))
        tauF = np.exp((vm + 158.6)/11.2) / (1.0 + np.exp((vm + 75.)/5.5))
        tauS = np.exp((vm + 183.6) / 15.24)
        df = (Hinf - f) / tauF
        ds = (Hinf - s) / tauS
        return [df*1e3, ds*1e3]


class LGNa(Channel):
    """Cortical sodium channel (Lewis & Gerstner 2002, p.124)
    """
    type = 'INa'
    
    def __init__(self, gbar=112.5*mS/cm**2, **kwds):
        init_state = OrderedDict([('m', 0.019), ('h', 0.876)]) 
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.erev = 74*mV
        
    def open_probability(self, state):
        return state[self, 'm']**3 * state[self, 'h']

    def derivatives(self, state):
        # temperature dependence of rate constants
        # TODO: not sure about the base temp:
        q10 = 3 ** ((self.sim.temp - 37.) / 10.)
        
        vm = state[self.section, 'V']
        m = state[self, 'm']
        h = state[self, 'h']

        #vm = vm + 65e-3   ## gating parameter eqns assume resting is 0mV
        vm *= 1000.   ##  ..and that Vm is in mV
        
        am = (-3020 + 40 * vm)  / (1.0 - np.exp(-(vm - 75.5) / 13.5))
        bm = 1.2262 / np.exp(vm / 42.248)
        mtau = 1 / (am + bm)
        minf = am * mtau
        dm = q10 * (minf - m) / mtau
        
        ah = 0.0035 / np.exp(vm / 24.186)
        # note: bh as originally written causes integration failures; we use
        # an equivalent expression that behaves nicely under floating point stress.
        #bh = (0.8712 + 0.017 * vm) / (1.0 - np.exp(-(51.25 + vm) / 5.2))
        bh = 0.017 * (51.25 + vm) / (1.0 - np.exp(-(51.25 + vm) / 5.2))
        htau = 1 / (ah + bh)
        hinf = ah * htau
        dh = q10 * (hinf - h) / htau

        return [dm*1e3, dh*1e3]


class LGKfast(Channel):
    """Cortical fast potassium channel (Lewis & Gerstner 2002, p.124)
    """
    type = 'IKf'
    
    def __init__(self, gbar=225*mS/cm**2, **kwds):
        init_state = OrderedDict([('n', 0.00024)]) 
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.erev = -90*mV
        
    def open_probability(self, state):
        return state[self, 'n']**2

    def derivatives(self, state):
        # temperature dependence of rate constants
        # TODO: not sure about the base temp:
        q10 = 3 ** ((self.sim.temp - 37.) / 10.)
        
        vm = state[self.section, 'V']
        n = state[self, 'n']

        #vm = vm + 65e-3   ## gating parameter eqns assume resting is 0mV
        vm *= 1000.   ##  ..and that Vm is in mV
        
        an = (vm - 95) / (1.0 - np.exp(-(vm - 95) / 11.8))
        bn = 0.025 / np.exp(vm / 22.22)
        ntau = 1 / (an + bn)
        ninf = an * ntau
        dn = q10 * (ninf - n) / ntau
        return [dn*1e3]


class LGKslow(Channel):
    """Cortical slow potassium channel (Lewis & Gerstner 2002, p.124)
    """
    type = 'IKs'
    
    def __init__(self, gbar=0.225*mS/cm**2, **kwds):
        init_state = OrderedDict([('n', 0.0005)]) 
        Channel.__init__(self, gbar=gbar, init_state=init_state, **kwds)
        self.erev = -90*mV
        
    def open_probability(self, state):
        return state[self, 'n']**4

    def derivatives(self, state):
        # temperature dependence of rate constants
        # TODO: not sure about the base temp:
        q10 = 3 ** ((self.sim.temp - 37.) / 10.)
        
        vm = state[self.section, 'V']
        n = state[self, 'n']

        #vm = vm + 65e-3   ## gating parameter eqns assume resting is 0mV
        vm *= 1000.   ##  ..and that Vm is in mV
        
        an = 0.014 * (vm + 44) / (1.0 - np.exp(-(44 + vm) / 2.3))
        bn = 0.0043 / np.exp((vm + 44) / 34)
        ntau = 1 / (an + bn)
        ninf = an * ntau
        dn = q10 * (ninf - n) / ntau

        return [dn*1e3]




# alpha synapse
#Alpha_t0 = 500.  # msec
#Alpha_tau = 2.0
#gAlpha = 1e-3 * Area/cm**2
#EAlpha = -7e-3  # V

# def IAlpha(Vm, t):
#     if t < Alpha_t0:
#         return 0.
#     else:
#         # g = gmax * (t - onset)/tau * exp(-(t - onset - tau)/tau)
#         tn = t - Alpha_t0
#         if tn > 10.0 * Alpha_tau:
#             return 0.
#         else:
#             return gAlpha * (Vm - EAlpha)*(tn/Alpha_tau) * np.exp(-(tn-Alpha_tau)/Alpha_tau)

