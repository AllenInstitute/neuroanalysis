import os
from typing import Literal

import h5py
import numpy as np
import pytest
from neuron import h

from neuroanalysis.data import TSeries, PatchClampRecording
from neuroanalysis.test_pulse import PatchClampTestPulse
from neuroanalysis.units import pA, mV, uV, MOhm, pF, uF, us, ms, cm, nA, um, mm


h.load_file('stdrun.hoc')


@pytest.mark.parametrize('pamp', [-100*pA, -11*pA, 12*pA])
@pytest.mark.parametrize('r_input', [103*MOhm, 200*MOhm, 499*MOhm])
@pytest.mark.parametrize('r_access', [5*MOhm, 10*MOhm, 15*MOhm])
@pytest.mark.parametrize('c_soma', [50*pF, 100*pF, 200*pF])
@pytest.mark.parametrize('c_pip', [1*pF, 3*pF, 10*pF])
# @pytest.mark.parametrize('only', ['input_resistance', 'capacitance', 'access_resistance'])
def test_ic_pulse(pamp, r_input, r_access, c_soma, c_pip, only=None):
    tp_kwds = dict(noise=10*uV, pamp=pamp, pdur=200*ms, mode='ic', r_access=r_access, r_input=r_input, c_soma=c_soma, c_pip=c_pip)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    if only:
        only = [only]
    check_analysis(tp, tp_kwds, only=only)


@pytest.mark.parametrize('pamp', [-85*mV, -75*mV, -55*mV])
@pytest.mark.parametrize('r_input', [100*MOhm, 200*MOhm, 500*MOhm])
@pytest.mark.parametrize('r_access', [5*MOhm, 10*MOhm, 15*MOhm])
@pytest.mark.parametrize('c_soma', [50*pF, 100*pF, 200*pF])
@pytest.mark.parametrize('c_pip', [1*pF, 3*pF, 10*pF])
# @pytest.mark.parametrize('only', ['input_resistance', 'capacitance', 'access_resistance'])
def test_vc_pulse(pamp, r_input, r_access, c_soma, c_pip, only=None):
    tp_kwds = dict(noise=1*pA, pamp=pamp, pdur=20*ms, mode='vc', hold=-65*mV, r_input=r_input, r_access=r_access, c_soma=c_soma, c_pip=c_pip)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    if only:
        only = [only]
    check_analysis(tp, tp_kwds, only=only)


def test_pulse_in_bath():
    tp_kwds = dict(noise=1e-13, pamp=-10*mV, mode='vc', c_soma=False, c_pip=3*pF, r_input=False, r_access=10*MOhm)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    assert np.isnan(tp.analysis['capacitance'])
    assert np.isclose(tp.analysis['steady_state_resistance'], tp_kwds['r_access'], rtol=0.3)

    tp_kwds = dict(noise=1e-6, pamp=-100*pA, pdur=100*ms, mode='ic', c_soma=False, c_pip=3*pF, r_input=False, r_access=10*MOhm)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    assert np.isnan(tp.analysis['capacitance'])
    assert np.isclose(tp.analysis['steady_state_resistance'], tp_kwds['r_access'], rtol=0.3)


def test_leaky_cell():
    tp_kwds = dict(noise=0, pamp=-10*mV, mode='vc', r_input=80*MOhm, rmp_soma=-30*mV)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    check_analysis(tp, tp_kwds)

    tp_kwds = dict(noise=0, pamp=-100*pA, pdur=200*ms, mode='ic', r_input=80*MOhm, rmp_soma=-30*mV)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    check_analysis(tp, tp_kwds)


def test_with_60Hz_noise():
    assert False  # TODO


def test_with_12kHz_noise():
    assert False  # TODO


def test_clogged_pipette_with_soma():
    shared_kwds = dict(noise=0, c_soma=80*pF, c_pip=3*pF, r_input=100*MOhm, r_access=50*MOhm)
    tp_kwds = dict(pamp=-100*pA, pdur=200*ms, mode='ic', **shared_kwds)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    check_analysis(tp, tp_kwds)

    tp_kwds = dict(pamp=-20*mV, pdur=10*ms, mode='vc', **shared_kwds)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    check_analysis(tp, tp_kwds)


def test_clogged_pipette_in_bath():
    tp_kwds = dict(noise=0, pamp=-10*mV, mode='vc', c_soma=False, c_pip=3*pF, r_input=False, r_access=30*MOhm)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    assert np.isnan(tp.analysis['capacitance'])
    assert np.isclose(tp.analysis['steady_state_resistance'], tp_kwds['r_access'], rtol=0.3)

    tp_kwds = dict(noise=0, pamp=-100*pA, pdur=200*ms, mode='ic', c_soma=False, c_pip=3*pF, r_input=False, r_access=30*MOhm)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    assert np.isnan(tp.analysis['capacitance'])
    assert np.isclose(tp.analysis['steady_state_resistance'], tp_kwds['r_access'], rtol=0.3)


def test_cell_attached():
    tp_kwds = dict(noise=0, pamp=-10*mV, mode='vc', c_soma=0.1*pF, c_pip=3*pF, r_input=1000*MOhm, r_access=10*MOhm)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    assert np.isclose(tp.analysis['steady_state_resistance'], tp_kwds['r_access'] + tp_kwds['r_input'], rtol=0.3)

    tp_kwds = dict(noise=0, pamp=-100*pA, pdur=200*ms, mode='ic', c_soma=0.1*pF, c_pip=3*pF, r_input=1000*MOhm, r_access=10*MOhm)
    tp, _ = create_mock_test_pulse(**tp_kwds)
    assert np.isclose(tp.analysis['steady_state_resistance'], tp_kwds['r_access'] + tp_kwds['r_input'], rtol=0.3)


def capacitance(section):
    """Return the capacitance of a soma in F."""
    # its units are (µF / cm²)
    return (section.cm * uF / cm ** 2) * section_surface(section)


def trunc_cone_surface_area(base_radius, tip_radius, length):
    return np.pi * (base_radius + tip_radius) * np.sqrt((base_radius - tip_radius)**2 + length**2)


def section_surface(section) -> float:
    """Return the surface area of a section (truncated cone) in m²."""
    # its units are um * um
    a_r = section(0).diam * um / 2
    b_r = section(1).diam * um / 2
    length = section.L * um
    return trunc_cone_surface_area(a_r, b_r, length)


def resistance(s):
    """Return the resistance of a soma in Ohms."""
    # its units are S/cm²
    return 1 / ((s(0.5).pas.g / cm**2) * section_surface(s))


def set_resistance(s, r):
    """Set the resistance of a soma in Ohms."""
    s(0.5).pas.g = 1 / (r / cm ** 2 * section_surface(s))


def set_pip_cap(v):
    # TODO we need to build https://www.neuron.yale.edu/phpBB/viewtopic.php?t=203
    # mech.c = h.Matrix(1, 1, 2).from_python([[v * uF]])
    cmat = h.Matrix(2, 2, 2).ident()
    cmat.setval(0, 1, v * uF)
    gmat = h.Matrix(2, 2, 2).ident()
    y = h.Vector(2)
    y0 = h.Vector(2)
    b = h.Vector(2)

    return h.LinearMechanism(cmat, gmat, y, y0, b), cmat, gmat, y, y0, b


def _make_ic_command(connection, amplitude, start, duration):
    ic = h.IClamp(connection)
    ic.amp = amplitude / nA
    ic.dur = duration / ms
    ic.delay = start / ms
    return ic


def create_mock_test_pulse(
        start: float = 5*ms,
        pdur: float = 200*ms,
        pamp: float = -10*pA,
        hold: float = 0.0,
        rmp_soma: float = -65*mV,
        mode: Literal['ic', 'vc'] = 'ic',
        dt: float = 10*us,
        r_access: float = 10*MOhm,
        r_input: float = 200*MOhm,
        c_soma: float = 100*pF,
        c_pip: float = 5*pF,
        plot: bool = False,
        noise: float = 0.5*pA,
        assert_valid: bool = False,
):
    settle = 500 * ms if mode == 'ic' else 50 * ms
    pulse = np.ones((int((settle + start + pdur + settle) // dt),)) * hold
    pulse[int((settle + start) // dt):int((settle + start + pdur) // dt)] = pamp

    pip_sections = _create_pipette(r_access, c_pip)

    if c_soma and r_input:
        soma = h.Section()
        soma.insert('pas')
        soma.cm = 1.0  # µF/cm²
        soma.L = soma.diam = (500 / np.pi) ** 0.5  # um (500 um²)
        soma.cm = soma.cm * c_soma / capacitance(soma)
        set_resistance(soma, r_input)
        pip_sections[-1].connect(soma(0.5), 1)
        for seg in soma:
            seg.pas.e = rmp_soma / mV
    else:
        # connect the pipette to ground
        ground = h.VClamp(pip_sections[-1](1))
        ground.dur[0] = 1e9  # clamp forever
        ground.amp[0] = 0

    clamp_connection = pip_sections[0](0)

    def run():
        vinit = -60  # mV

        h.init()
        h.finitialize(vinit)

        h.dt = dt / ms
        h.continuerun((settle + start + pdur + settle) / ms)

    if mode == 'ic':
        pre_ic = _make_ic_command(clamp_connection, hold, 0, settle + start)
        pulse_ic = _make_ic_command(clamp_connection, pamp, settle + start, pdur)
        post_ic = _make_ic_command(clamp_connection, hold, settle + start + pdur, settle)

        pip_rec0 = h.Vector()
        pip_rec0.record(clamp_connection._ref_v)

        run()
        out = pip_rec0.as_numpy() * mV
    else:
        vc = h.SEClamp(clamp_connection)
        vc.rs = 0.01 / MOhm  # just get out of the way of our segmented pipette

        vc.dur1 = (settle + start) / ms
        vc.amp1 = hold / mV
        vc.dur2 = pdur / ms
        vc.amp2 = pamp / mV
        vc.dur3 = settle / ms
        vc.amp3 = hold / mV

        vc_rec = h.Vector()
        vc_rec.record(vc._ref_i)

        run()
        out = vc_rec.as_numpy() * nA

    out = out[int(settle // dt):int((settle + start + pdur + settle) // dt)]
    if noise:
        out += np.random.normal(0, noise, out.shape)
    pulse = pulse[int(settle // dt):int((settle + start + pdur + settle) // dt)]

    tp = PatchClampTestPulse(PatchClampRecording(
        channels={
            'primary': TSeries(out, dt=dt),
            'command': TSeries(pulse, dt=dt)},
        dt=dt,
        t0=0,
        clamp_mode=mode,
        bridge_balance=0,
        lpf_cutoff=None,
        pipette_offset=0,
        holding_current=hold if mode == 'ic' else None,
        holding_potential=hold if mode == 'vc' else None,
    ))
    if plot:
        tp.plot()
        # pg.plot(pulse, title=f'{mode} command')
    if assert_valid:
        try:
            check_analysis(tp, locals())
        except AssertionError as e:
            print("assertion failed", e)
    return tp, locals()  # NEURON blows up if GC deletes objects before we're done


def _create_pipette(r_access, c_pip):
    pipette_halfangle = np.deg2rad(20 / 2)
    base_radius = 0.5 * 0.86 * mm
    tip_radius = 0.5 * 1.0 * um
    length = base_radius / np.tan(pipette_halfangle)
    resistivity = np.pi * base_radius * tip_radius * r_access / length
    n_pip_sections = 15
    pip_sections = []
    # make a series of connected sections to approximate a truncated conic conductor.
    # sections will have progressively smaller radius. section lengths are chosen
    # such that all sections have equal resistance given a constant resistivity.
    axial_resistance_per_section = r_access / n_pip_sections
    next_radius = base_radius
    for i in range(n_pip_sections):
        r = next_radius

        # solve for length of truncated cone with specified resistance
        # R = ρ * l / (pi * base_r * tip_r)
        # l = (pi * base_r * tip_r) * R / ρ
        # tip_r = base_r - l * tan(halfangle)
        # l = (pi * base_r * (base_r - l * tan(halfangle))) * R / ρ
        # l * ρ / (pi * base_r * R) = base_r - l * tan(halfangle)
        # l * ρ / (pi * base_r * R) + l * tan(halfangle) = base_r
        # l * (ρ / (pi * base_r * R) + tan(halfangle)) = base_r
        # l = base_r / (ρ / (pi * base_r * R) + tan(halfangle))
        l = r / (resistivity / (np.pi * r * axial_resistance_per_section) + np.tan(pipette_halfangle))

        # now choose an effective cylinder radius that gives the same resistance
        # R = ρ * l / A
        # A = ρ * l / R  = pi * r^2
        # r = sqrt((ρ * l) / (pi * R))
        cyl_r = np.sqrt((resistivity * l) / (np.pi * axial_resistance_per_section))

        sec = h.Section()
        sec.nseg = 1
        sec.L = l / um
        sec.diam = 2 * cyl_r / um
        sec.Ra = resistivity / cm
        if i > 0:
            pip_sections[-1].connect(sec(0), 1)
        pip_sections.append(sec)
        next_radius -= l * np.tan(pipette_halfangle)

    total_surface_area = np.sum([section_surface(sec) for sec in pip_sections])
    pip_cap_per_area = c_pip / total_surface_area
    for sec in pip_sections:
        sec.cm = pip_cap_per_area * cm ** 2 / uF
    return pip_sections


def expected_testpulse_values(tp_kwds):
    values = {
        'access_resistance': tp_kwds.get('r_access', 10*MOhm),
        'capacitance': tp_kwds.get('c_soma', 100*pF),
        'input_resistance': tp_kwds.get('r_input', 200*MOhm),
    }
    if tp_kwds.get('mode', 'ic') == 'ic':
        # values['baseline_potential'] = 0  # TODO
        values['baseline_current'] = tp_kwds.get('hold', 0)
    else:
        values['baseline_potential'] = tp_kwds.get('hold', 0)
        # values['baseline_current'] = 0  # TODO

    return values


def check_analysis(pulse, tp_kwds, only=None, tol_override=None):
    measured = pulse.analysis
    expected = expected_testpulse_values(tp_kwds)
    
    # how much error should we tolerate for each parameter?
    err_tolerance = {
        'baseline_potential': (0.01, 5*mV),
        'baseline_current': (0.01, 1e-12),
        'access_resistance': (0.3, 5e4),
        'input_resistance': (0.1, 1e5),
        'capacitance': (0.3, 1e-13),
    }
    if tol_override:
        err_tolerance.update(tol_override)
    mistakes = False
    if only:
        expected = {k: v for k, v in expected.items() if k in only}
    for k, v1 in expected.items():
        v2 = measured[k]
        if v1 is None:
            print(f"FAILURE: expected None for {k}, measured {v2}")
            mistakes = mistakes or v2 is not None
            continue
        elif v2 is None:
            print(f"FAILURE: expected {v1} for {k}, measured None")
            mistakes = True
            continue
        abs_err = abs(v1 - v2)
        if v1 == 0:
            rel_err = abs_err
        else:
            rel_err = abs((v1 - v2) / v1)
        rtol, atol = err_tolerance[k]
        if rel_err > rtol and abs_err > atol:
            print(f"FAILURE: expected {v1:g} for {k}, got {v2:g} (rel_err {rel_err:g} > {rtol :g}) "
                  f"(abs_err {abs_err:g} > {atol:g})")
            mistakes = True
        else:
            print(f"success: expected {v1:g} for {k}, got {v2:g} (rel_err {rel_err:g}) (abs_err {abs_err:g})")
    assert not mistakes, ', '.join((f'{k}={v}' for k, v in tp_kwds.items()))


def test_load():
    tp, _ = create_mock_test_pulse()
    new_tp = PatchClampTestPulse.load(tp.save())
    for k, v in tp.analysis.items():
        if isinstance(v, (np.ndarray, int, float)):
            assert np.allclose(v, new_tp.analysis[k])
        else:
            assert v == new_tp.analysis[k]


def test_append_stack():
    from neuroanalysis.test_pulse_stack import H5BackedTestPulseStack

    f = h5py.File('test.h5', 'w')
    try:
        stack = H5BackedTestPulseStack(f.create_group('test_pulses'))
        tp = create_mock_test_pulse()[0]
        stack.append(tp, retain_data=False)
        assert len(stack) == 1
        assert stack[tp.recording.start_time].analysis == tp.analysis
    finally:
        f.close()
        os.remove('test.h5')


def test_bath_ugly():
    from neuroanalysis.test_pulse_stack import H5BackedTestPulseStack

    fn = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'test_data', 'bath-ugly.h5')
    print(fn)
    f = h5py.File(fn, 'r')
    tps = H5BackedTestPulseStack(f['test_pulses'])
    assert len(tps) == 1
    tp = tps.at_time(float('inf'))
    assert tp.analysis
    assert np.isnan(tp.analysis['capacitance'])
    assert tp.analysis['steady_state_resistance'] < 10e6


def main():
    import pyqtgraph as pg
    from pyqtgraph.parametertree import ParameterTree, interact

    global params, app, tree
    params = interact(
        create_mock_test_pulse,
        rmp_soma={'siPrefix': True, 'suffix': 'V'},
        r_access={'siPrefix': True, 'suffix': 'Ω'},
        r_input={'siPrefix': True, 'suffix': 'Ω'},
        c_soma={'siPrefix': True, 'suffix': 'F'},
        c_pip={'siPrefix': True, 'suffix': 'F'},
        pamp={'siPrefix': True, 'suffix': 'V/A'},
        hold={'siPrefix': True, 'suffix': 'V/A'},
    )
    app = pg.mkQApp()
    tree = ParameterTree()
    tree.setParameters(params)
    tree.show()
    pg.exec()
    failures = [
        "dict(pamp=-0.02, pdur=0.01, mode='vc', noise=0, c_soma=8e-11, c_pip=3e-12, r_input=100e6, r_access=100e6)",
        # clogged pipette
        "dict(noise=0, pamp=-0.01, mode='vc', c_soma=1e-13, c_pip=3e-12, r_input=1000e6, r_access=10e6)",  # attached
        "dict(noise=10e-06, pamp=1.2e-11, pdur=0.2, mode='ic', r_access=5e6, r_input=103e6, c_soma=5e-11, c_pip=1e-11)",
        "dict(noise=10e-06, pamp=-1.1e-11, pdur=0.2, mode='ic', r_access=5e6, r_input=200e6, c_soma=5e-11, c_pip=1e-11)",
        "dict(noise=10e-06, pamp=-1e-10, pdur=0.2, mode='ic', r_access=15e6, r_input=103e6, c_soma=5e-11, c_pip=1e-11)",
        "dict(noise=10e-06, pamp=-1e-10, pdur=0.2, mode='ic', r_access=15e6, r_input=499e6, c_soma=5e-11, c_pip=1e-11)",
        "dict(noise=10e-06, pamp=-1.1e-11, pdur=0.2, mode='ic', r_access=15e6, r_input=499e6, c_soma=5e-11, c_pip=1e-11)",
        "dict(noise=10e-06, pamp=1.2e-11, pdur=0.2, mode='ic', r_access=10e6, r_input=200e6, c_soma=1e-10, c_pip=1e-11)",
    ]
    for _kwds in failures:
        title = _kwds[5:-1]
        _kwds = eval(_kwds)
        _tp, vc_locals = create_mock_test_pulse(**_kwds)
        print(title)
        print(_tp.recording.clamp_mode, _tp.plot_units, _tp.analysis['time_constant'])
        plt = _tp.plot()
        plt.setTitle(f"tau: {_tp.analysis['time_constant']}, nrmse: {_tp.main_fit_result['nrmse']}")
        print(f"tp.analysis: {_tp.analysis}")
        pg.exec()
        check_analysis(_tp, _kwds)


if __name__ == '__main__':
    main()
