import numpy as np
from . import Sim, Section, Leak, LGKfast, LGKslow, LGNa, Noise, PatchClamp
from ..data import TSeries, PatchClampRecording
from ..units import mS, uV, mV, pA, cm, MOhm, ms


class ModelCell(object):
    """A simulated patch-clamped neuron for generating test data.
    """
    def __init__(self):
        self.sim = Sim()
        self._is_settled = False
        
        # Add noise to recording
        self.recording_noise = True
        self.rec_noise_sigma = {'ic': 50*uV, 'vc': 5*pA}

        # Create a single compartment
        radius = (5e-10 / (4*np.pi)) ** 0.5
        self.soma = Section(name='soma', radius=radius)
        self.sim.add(self.soma)

        # Add channels to the membrane
        self.mechs = {
            'leak': Leak(gbar=1.0*mS/cm**2, erev=-75*mV),
            'leak0': Leak(gbar=0, erev=0),  # simulate dying cell
            'lgkfast': LGKfast(gbar=225*mS/cm**2),
            'lgkslow': LGKslow(gbar=0.225*mS/cm**2),
            'lgkna': LGNa(),
            'noise': Noise(),  # adds realism, but slows down the integrator a lot.
        }
        for m in self.mechs.values():
            self.soma.add(m)

        # Add a patch clamp electrode
        self.clamp = PatchClamp(name='electrode', mode='ic', ra=10*MOhm)
        self.clamp.set_holding('vc', -75*mV)
        self.soma.add(self.clamp)
        
    def enable_mechs(self, mechs):
        """Enable a specific set of mechanisms; disable all others.
        """
        for mech in self.mechs.values():
            mech.enabled = False
        for mech in mechs:
            self.mechs[mech].enabled = True

    def test(self, command, mode):
        """Send a command (TSeries) to the electrode and return a PatchClampRecording
        that contains the result.
        """
        self.clamp.set_mode(mode)
        self.sim.dt = command.dt
        self._is_settled = False
        
        self.settle()
        
        # run simulation
        self.clamp.queue_command(command.data, command.dt)
        result = self.sim.run(len(command))
        
        # collect soma and pipette potentials
        t = result['t']
        vm = result['soma.V']
        response = TSeries(vm, time_values=t)
        pip = result['electrode.V'] if mode == 'ic' else result['electrode.I']
        
        # Add in a little electrical recording noise        
        if self.recording_noise:
            enoise = self.rec_noise_sigma[mode]
            pip = pip + np.random.normal(size=len(pip), scale=enoise)
        
        recording = TSeries(pip, time_values=t)
        
        channels = {'command': command, 'primary': recording, 'vsoma': response}
        kwds = {
            'clamp_mode': mode,
            'bridge_balance': 0,
            'lpf_cutoff': None,
            'pipette_offset': 0,
        }
        if mode == 'ic':
            kwds['holding_current'] = self.clamp.holding['ic']
        elif mode == 'vc':
            kwds['holding_potential'] = self.clamp.holding['vc']
            
        return PatchClampRecording(channels=channels, **kwds)
    
    def settle(self, t=1.0):
        """Run the simulation with no input to let it settle into steady state.
        """
        if self._is_settled:
            return
        n = int(t / self.sim.dt)
        noise_enabled = self.mechs['noise'].enabled
        self.mechs['noise'].enabled = False
        self.sim.run(n, hmax=1*ms)
        self.mechs['noise'].enabled = noise_enabled
        self._is_settled = True
        
    def input_resistance(self):
        self.settle()
        return 1.0 / self.soma.conductance(self.sim.last_state)
    
    def capacitance(self):
        return self.soma.cap

    def resting_potential(self):
        self.clamp.set_mode('ic')
        self.settle()
        return self.sim.last_state['electrode.V']

    def resting_current(self):
        self.clamp.set_mode('vc')
        self.settle()
        return self.sim.last_state['electrode.I']
