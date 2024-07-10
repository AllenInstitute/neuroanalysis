import numpy as np
from neuron import h
import pyqtgraph as pg

from neuroanalysis.data import TSeries, PatchClampRecording
from neuroanalysis.test_pulse import PatchClampTestPulse
from neuroanalysis.units import pA, mV, MOhm, pF, uF, us, ms, cm, nA, um


def test_ic_pulse():
    # Just test against a simple R/C circuit attached to a pipette
    tp_kwds = dict(pamp=-100*pA, mode='ic', r_access=10*MOhm)
    tp = create_test_pulse(**tp_kwds)
    check_analysis(tp, soma, tp_kwds)

def test_vc_pulse():
    # Just test against a simple R/C circuit attached to a pipette
    tp_kwds = dict(pamp=-85*mV, mode='vc', hold=-65*mV)
    tp = create_test_pulse(**tp_kwds)
    check_analysis(tp, soma, tp_kwds)


def test_insignificant_transient():
    tp_kwds = dict(pamp=-10*mV, mode='vc')
    tp = create_test_pulse()
    # prevent the cell from impacting the pulse
    check_analysis(tp, soma, tp_kwds)
    assert False  # TODO


def test_with_60Hz_noise():
    assert False  # TODO


def capacitance(s):
    """Return the capacitance of a soma in F."""
    # its units are (µF / cm²)
    return (s.cm * uF / cm**2) * surface_area(s)


def surface_area(s) -> float:
    """Return the surface area of a soma in m²."""
    # its units are µm * µm
    return (s.L * um) * (s.diam * um) * np.pi


def resistance(s):
    """Return the resistance of a soma in Ohms."""
    # its units are S/cm²
    return 1 / ((s(0.5).pas.g / cm**2) * surface_area(s))


def set_resistance(s, r):
    """Set the resistance of a soma in Ohms."""
    s(0.5).pas.g = 1 / (r / cm**2 * surface_area(s))


# TODO weird stuff with reuse of these; maybe make new ones each time
soma = h.Section()
soma.insert('pas')
soma.cm = 1.0  # capacitance in µF/cm²
soma.L = soma.diam = (500 / np.pi) ** 0.5  # µm

vc = h.SEClamp(soma(0.5))
vc_rec = h.Vector()
vc_rec.record(vc._ref_i)

# TODO we need to build https://www.neuron.yale.edu/phpBB/viewtopic.php?t=203
ic_rec = h.Vector()
ic_rec.record(soma(0.5)._ref_v)

h.load_file('stdrun.hoc')


def _make_ic_command(amplitude, duration, start):
    ic = h.IClamp(soma(0.5))
    ic.amp = amplitude / nA
    ic.dur = duration / ms
    ic.delay = start / ms
    return ic


def create_test_pulse(start=5*ms, pdur=10*ms, pamp=-10*pA, hold=0, mode='ic', dt=10*us, r_access=10*MOhm, r_input=100*MOhm, c_soma=25*pF, noise=5*pA):
    ic_rec.clear()
    vc_rec.clear()
    settle = 50 * ms

    if mode == 'ic':
        _make_ic_command(hold, settle + start, 0)
        _make_ic_command(pamp, pdur, settle + start)
        _make_ic_command(hold, settle, settle + start + pdur)
    else:
        vc.dur1 = (settle + start) / ms
        vc.amp1 = hold / mV
        vc.dur2 = pdur / ms
        vc.amp2 = pamp / mV
        vc.dur3 = settle / ms
        vc.amp3 = hold / mV

    vinit = -65  # mV
    vc.rs = r_access / MOhm  # Rs, in MOhms
    soma.cm = c_soma / capacitance(soma)
    set_resistance(soma, r_input)

    h.init()
    h.finitialize(vinit)

    h.dt = dt / ms
    h.continuerun((settle + start + pdur + settle) / ms)

    if mode == 'ic':
        out = ic_rec.as_numpy() * mV
        cmd_label = 'A'
        label = 'V'
    else:
        out = vc_rec.as_numpy() * nA
        # not recorded by NEURON, so we just fake it.
        cmd_label = 'V'
        label = 'A'

    pulse = np.ones_like(out) * hold
    pulse[int((settle + start) // dt):int((settle + start + pdur) // dt)] = pamp
    out = out[int(settle // dt):int((settle + start + pdur + settle) // dt)]
    pulse = pulse[int(settle // dt):int((settle + start + pdur + settle) // dt)]

    pg.plot(pulse, labels={'left': ('Command', cmd_label), 'bottom': ('time', 's')})
    pg.plot(out, labels={'left': ('Response', label), 'bottom': ('time', 's')})
    pg.exec()

    return PatchClampTestPulse(
        PatchClampRecording(
            channels={
                'primary': TSeries(out, dt=dt),
                'command': TSeries(pulse, dt=dt)},
            dt=dt,
            t0=0,
            clamp_mode=mode,
            bridge_balance=0,
            lpf_cutoff=None,
            pipette_offset=0,
        ),
    )


def expected_testpulse_values(cell, tp_kwds):
    values = {
        'access_resistance': vc.rs * MOhm,
        'capacitance': capacitance(cell),
        'input_resistance': resistance(soma),
    }
    if tp_kwds.get('mode', 'ic') == 'ic':
        values['baseline_potential'] = 0  # TODO
        values['baseline_current'] = tp_kwds.get('hold', 0)
    else:
        values['baseline_potential'] = tp_kwds.get('hold', 0)
        values['baseline_current'] = 0  # TODO

    return values


def check_analysis(pulse, cell, tp_kwds):
    measured = pulse.analysis
    expected = expected_testpulse_values(cell, tp_kwds)
    
    # how much error should we tolerate for each parameter?
    err_tolerance = {
        'baseline_potential': 0.01,
        'baseline_current': 0.01,
        'access_resistance': 0.3,
        'input_resistance': 0.1,
        'capacitance': 0.3,
    }
    mistakes = False
    for k, v1 in expected.items():
        v2 = measured[k]
        if v1 is None:
            print(f"Expected None for {k}, measured {v2}")
            continue
        elif v2 is None:
            print(f"Expected {v1} for {k}, measured None")
            continue
        if v2 == 0:
            err = abs(v1 - v2)
        else:
            err = abs((v1 - v2) / v2)
        if err > err_tolerance[k]:
            print(f"Expected {v1:g} for {k}, got {v2:g} (err {err:g} > {err_tolerance[k]:g})")
            mistakes = True
    assert not mistakes


def example(mode='ic', holding=0.05):
    # Step 1: Create a cell model
    soma = h.Section(name='soma')

    # Insert passive properties
    soma.insert('pas')

    # Step 2: Insert a current clamp for the holding current
    pre_ic = h.IClamp(soma(0.5))
    pre_ic.delay = 0  # Start immediately
    pre_ic.dur = 50  # Duration long enough to settle before test pulse
    pre_ic.amp = holding  # Holding current amplitude in nA

    # Step 3: Insert a current clamp for the test pulse
    pulse_ic = h.IClamp(soma(0.5))
    pulse_ic.delay = 50  # Start after holding current has settled
    pulse_ic.dur = 1  # Duration of the test pulse
    pulse_ic.amp = 0.1  # Test pulse amplitude in nA

    # Step 3.5: Insert a current clamp for the holding current
    post_ic = h.IClamp(soma(0.5))
    post_ic.delay = 51  # Start after the pulse
    post_ic.dur = 50  # Duration long enough to settle before test pulse
    post_ic.amp = holding  # Holding current amplitude in nA

    # Step 4: Set up recording vectors
    v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
    t = h.Vector().record(h._ref_t)  # Time vector
    pre_i = h.Vector().record(pre_ic._ref_i)  # Holding current vector
    post_i = h.Vector().record(post_ic._ref_i)  # Holding current vector
    i_test = h.Vector().record(pulse_ic._ref_i)  # Test pulse current vector

    # Step 5: Run the simulation
    h.finitialize(-65)
    h.continuerun(101)  # Run long enough to cover both holding and test pulse phases

    # Step 6: Convert to NumPy arrays
    v_numpy = np.array(v)
    t_numpy = np.array(t)
    pre_i_np = np.array(pre_i)
    post_i_np = np.array(post_i)
    i_test_numpy = np.array(i_test)

    # Print or analyze the results
    print("Time (ms):", t_numpy)
    print("Membrane potential (mV):", v_numpy)
    print("Holding current (nA):", pre_i_np)
    print("Test pulse current (nA):", i_test_numpy)
    pg.plot(t_numpy, v_numpy * mV, title='Membrane potential', labels={'left': ('Vm', 'V'), 'bottom': ('time', 'ms')})
    pg.plot(t_numpy, (pre_i_np + i_test_numpy + post_i_np) * nA, title='Command', labels={'left': ('I', 'A'), 'bottom': ('time', 'ms')})
    return v_numpy, t_numpy, pre_i_np, i_test_numpy


if __name__ == '__main__':
    # v_numpy, t_numpy, i_holding_numpy, i_test_numpy = example()
    kwds = dict(pamp=100*pA, mode='ic', r_access=100*MOhm, hold=10*pA)
    tp = create_test_pulse(**kwds)
    tp.plot()
    pg.exec()
    check_analysis(tp, soma, kwds)

    # print("Vm %g mV    Rm %g MOhm" % (model_cell.resting_potential()*1000, model_cell.input_resistance()/MOhm))

    kwds = dict(pamp=-80*mV, mode='vc', r_access=15*MOhm, hold=-65*mV)
    tp = create_test_pulse(**kwds)
    tp.plot()
    pg.exec()
    check_analysis(tp, soma, kwds)
