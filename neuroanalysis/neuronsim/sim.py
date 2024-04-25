# -*- coding: utf-8 -*-
"""
Simple neuron simulator for Python.
Also simulates voltage clamp and current clamp with access resistance.

Luke Campagnola 2015
github.com/campagnola/neurodemo
"""

from collections import OrderedDict
import numpy as np
import scipy.integrate
from ..units import us, ms


class SimState(object):
    """Contains the state of all diff. eq. variables in the simulation.

    During simulation runs, this is used to carry information about all
    variables at the current timepoint. After the simulation finishes, this is
    used to carry all state variable data collected during the simulation.

    Parameters
    ==========
        difeq_vars: list
            Names of all diff. eq. state variables
        dep_vars: dict
            Name:function pairs for all dependent variables that may be computed
        difeq_state: list
            Initial values for all dif. eq. state variables
        extra:
            Extra name:value pairs that may be accessed from this object
    """

    def __init__(self, difeq_vars, dep_vars=None, difeq_state=None, integrator='odeint', **extra):
        self.difeq_vars = difeq_vars
        # record indexes of difeq vars for fast retrieval
        self.indexes = dict([(k, i) for i, k in enumerate(difeq_vars)])

        self.dep_vars = dep_vars
        self.state = difeq_state

        self.extra = extra
        self.integrator = integrator

    def set_state(self, difeq_state):
        self.state = difeq_state

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.get_slice(key)
        # allow lookup by (object, var)
        if isinstance(key, tuple):
            key = f"{key[0].name}.{key[1]}"
        try:
            # try this first for speed
            return self.state[self.indexes[key]]
        except KeyError:
            if key in self.dep_vars:
                return self.dep_vars[key](self)
            else:
                return self.extra[key]

    def keys(self):
        return self.difeq_vars + list(self.extra.keys())
        # return list(self.indexes.keys()) + list(self.dep_vars.keys()) + list(self.extra.keys())

    def __contains__(self, key):
        # allow lookup by (object, var)
        if isinstance(key, tuple):
            key = f"{key[0].name}.{key[1]}"
        return key in self.indexes or key in self.dep_vars or key in self.extra

    def __str__(self):
        rep = f"SimState {id(self)}:\n"
        if self.state is not None:
            for i, k in enumerate(self.difeq_vars):
                rep += f"  {k} = {self.state[i][-1]}\n"
        else:
            rep += "  (no state)\n"
        return rep

    def get_final_state(self):
        """Return a dictionary of all diff. eq. state variables and dependent
        variables for all objects in the simulation.
        """
        return self.get_state_at_index(-1)

    def get_state_at_time(self, t):
        index = np.searchsorted(self['t'], t)
        return self.get_state_at_index(index)

    def get_state_at_index(self, index):
        s = self.copy()
        clip = not np.isscalar(self["t"])
        if clip:
            # only get results for the last timepoint
            s.set_state(self.state[:, index])

        state = {}
        for k in self.difeq_vars:
            state[k] = s[k]
        for k in self.dep_vars:
            state[k] = s[k]
        for k, v in self.extra.items():
            if clip:
                state[k] = v[index]
            else:
                state[k] = v

        return state

    def get_slice(self, sl):
        kwds = {'difeq_state': self.state[:, sl]}
        for k, v in self.extra.items():
            kwds[k] = v[sl]
        return self.copy(**kwds)

    def copy(self, **kwds):
        default_kwds = {
            'difeq_vars': self.difeq_vars,
            'dep_vars': self.dep_vars,
            'difeq_state': self.state,
            'integrator': self.integrator,
        }
        default_kwds.update(self.extra)
        default_kwds.update(kwds)
        return SimState(**default_kwds)


class Sim(object):
    """Simulator for a collection of objects that derive from SimObject
    """
    def __init__(self, objects=None, temp=37.0, dt=10*us, integrator:str='solve_ivp'):
        self._objects = []
        self._all_objs = None
        self._time = 0.0
        self.temp = temp
        self.dt = dt
        self.integrator = integrator
        self._simstate = None
        if objects is not None:
            for obj in objects:
                self.add(obj)

    def set_integrator(self, integrator:str):
        if integrator not in {"odeint", "solve_ivp"}:
            raise ValueError(f"Unknown integrator: {integrator}")
        self.integrator = integrator

    def change_dt(self, newdt:float=100e-6):
        newdt = np.clip(newdt, 5*us, 1*ms)
        self.dt = newdt
    
    def add(self, obj):
        assert obj._sim is None
        obj._sim = self
        self._objects.append(obj)
        return obj

    def all_objects(self):
        """Ordered dictionary of all objects to be simulated, keyed by their names.
        """
        if self._all_objs is None:
            objs = OrderedDict()
            for o in self._objects:
                if not o.enabled:
                    continue
                for k, v in o.all_objects().items():
                    if k in objs:
                        raise NameError(f'Multiple objects with same name "{k}": {objs[k]}, {v}')
                    objs[k] = v
            self._all_objs = objs
        return self._all_objs
    
    @property
    def time(self):
        return self._time

    def run(self, samples:int=1000, **kwds) -> SimState:
        """Run the simulation until a number of *samples* have been acquired.

        Extra keyword arguments are passed to `scipy.integrate.odeint()`.
        """
        # print("Integrator: ", self.integrator)
        # reset all_objs cache in case some part of the sim has changed
        self._all_objs = None
        all_objs = self.all_objects().values()

        # check that there is something to simulate
        if len(all_objs) == 0:
            raise RuntimeError("No objects added to simulation.")

        # Collect / prepare state variables for integration
        init_state = []
        difeq_vars = []
        dep_vars = {}
        for o in all_objs:
            pfx = f'{o.name}.'
            for k, v in o.difeq_state().items():
                difeq_vars.append(pfx + k)
                init_state.append(v)
            for k, v in o.dep_state_vars.items():
                dep_vars[pfx + k] = v
        self._simstate = SimState(difeq_vars, dep_vars)
        t = np.arange(0, samples) * self.dt + self._time

        opts = {"rtol": 1e-6, "atol": 1e-8, "hmax": 5e-4, "full_output": 1}
        opts.update(kwds)

        if self.integrator == 'odeint':
            # Run the simulation
            result, info = scipy.integrate.odeint(self.derivatives, init_state, t, tfirst=True, **opts)

            # Update current state variables
            p = 0
            for o in all_objs:
                nvar = len(o.difeq_state())
                o.update_state(result[-1, p : p + nvar])
                p += nvar
            self._time = t[-1]
            # print(f"   {self.integrator:s}  final state = {str(result.T[:, -1]):s}")
            # print("   start, finished at : ", t[0],t[-1])
            # print("   np.min(result.T): ", np.min(result.T), np.max(result.T))
            return SimState(difeq_vars, dep_vars, result.T, integrator=self.integrator, t=t)

        elif self.integrator == 'solve_ivp':
            """Notes:
            Different integrators were tried. 
            LSODA works ok; RK's seem to stall on AP. 
            Radau and BDF are very fast, but take some large steps, so the calculation of where the 
            pulse array is in the derivative via get_cmd are not always correct -
            the algorithm might step a long time into the future, invalidating all the
            commands in the queue, which are removed. Thus, only the first cmd
            is executed.
            Probably should not pop the queue in get_cmd until we are certain at THIS
            level (or maybe in runner?) that the trigger arrays are actually finished. 
            """
            # in future we will need to implement this instead: 
            result = scipy.integrate.solve_ivp(
                self.derivatives,
                t_span=(t[0], t[-1]),
                t_eval=t,
                y0=init_state,
                method="LSODA",  # runs ok with LSODA

                dense_output=False,
                # args=dep_vars,
                rtol = opts['rtol'], #**opts,
                atol = opts['atol'],
                max_step = opts['hmax'],
            )
            # Update current state variables
            p = 0
            for o in all_objs:
                nvar = len(o.difeq_state())
                # print("solve ivp state: ", p, nvar, result.y[p:p+nvar, -1])
                o.update_state(result.y[p:p+nvar, -1])
                p += nvar
            self._time = t[-1]
            # print(f"\n   {self.integrator:s}  {str(result.y[:, -1]):s}")
            # print("   start, finished at : ", t[0],t[-1])
            # print("    np.min(result.y): ", np.min(result.y), np.max(result.y))

            return SimState(difeq_vars, dep_vars, result.y, integrator=self.integrator, t=t)

    def derivatives(self, t, state):
        objs = self.all_objects().values()

        # bug: the integrators may trash their outputs later;
        # copy them before that can happen
        self._simstate.state = state.copy()

        self._simstate.extra['t'] = t
        d = []
        for o in objs:
            d.extend(o.derivatives(self._simstate))
        return d

    @property
    def last_state(self):
        """Return the last values of all state variables in a SimState object.
        """
        return self._simstate
    
    def state(self):
        """Return dictionary of all dependent and independent state
        variables.
        """
        state = {}
        for o in self.all_objects():
            for k, v in o.state(self._simstate).items():
                state[k] = v
        return state


class SimObject(object):
    """
    Base class for objects that participate in integration by providing a set
    of state variables and their derivatives.
    """

    instance_count = 0

    def __init__(self, init_state, name=None):
        self._sim = None
        if name is None:
            i = self.instance_count
            type(self).instance_count = i + 1
            if i == 0:
                name = self.type
            else:
                name = self.type + "%d" % i
        self._name = name
        self.enabled = True
        self._init_state = init_state.copy()  # in case we want to reset
        self._current_state = init_state.copy()
        self._sub_objs = []
        self.records = []
        self._rec_dtype = [(sv, float) for sv in init_state.keys()]

        # maps name:function for computing state vars that can be computed from
        # a SimState instance.
        self.dep_state_vars = {}

    @property
    def name(self):
        return self._name

    def all_objects(self):
        """SimObjects are organized in a hierarchy. This method returns an ordered
        dictionary of all enabled SimObjects in this branch of the hierarchy, beginning
        with self.
        """
        objs = OrderedDict()
        objs[self.name] = self
        for o in self._sub_objs:
            if not o.enabled:
                continue
            objs.update(o.all_objects())
        return objs

    def difeq_state(self):
        """An ordered dictionary of all variables required to solve the
        diff. eq. for this object.
        """
        return self._current_state

    def update_state(self, result):
        """Update diffeq state variables with their last simulated values.
        These will be used to initialize the solver when the next simulation
        begins.
        """
        for i, k in enumerate(self._current_state.keys()):
            self._current_state[k] = result[i]

    def derivatives(self, state):
        """Return derivatives of all state variables.

        Must be reimplemented in subclasses. This is used by the ODE solver
        to integrate during the simulation; should be as fast as possible.
        """
        raise NotImplementedError()

    @property
    def sim(self):
        """The Sim instance in which this object is being used."""
        return self._sim

