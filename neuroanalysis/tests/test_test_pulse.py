import numpy as np
import pytest
from neuron import h
import pyqtgraph as pg

from neuroanalysis.data import TSeries, PatchClampRecording
from neuroanalysis.test_pulse import PatchClampTestPulse
from neuroanalysis.units import pA, mV, MOhm, pF, uF, us, ms, cm, nA, um, mm

h.load_file('stdrun.hoc')


@pytest.mark.parametrize('pamp', [-100*pA, -10*pA, 10*pA])
@pytest.mark.parametrize('r_input', [100*MOhm, 200*MOhm, 500*MOhm])
@pytest.mark.parametrize('r_access', [5*MOhm, 10*MOhm, 15*MOhm])
@pytest.mark.parametrize('c_soma', [10*pF, 25*pF, 100*pF])
@pytest.mark.parametrize('only', ['access_resistance', 'capacitance', 'input_resistance', 'baseline_current'])
def test_ic_pulse(pamp, r_input, r_access, c_soma, only):
    tp_kwds = dict(pamp=pamp, mode='ic', r_access=r_access, r_input=r_input, c_soma=c_soma)
    tp, _ = create_test_pulse(**tp_kwds)
    check_analysis(tp, _['soma'], tp_kwds, only=[only])


@pytest.mark.parametrize('pamp', [-100*mV, -50*mV, -10*mV])
@pytest.mark.parametrize('r_input', [100*MOhm, 200*MOhm, 500*MOhm])
@pytest.mark.parametrize('r_access', [5*MOhm, 10*MOhm, 15*MOhm])
@pytest.mark.parametrize('c_soma', [10*pF, 25*pF, 100*pF])
@pytest.mark.parametrize('only', ['access_resistance', 'capacitance', 'input_resistance', 'baseline_potential'])
def test_vc_pulse(pamp, r_input, r_access, c_soma, only):
    tp_kwds = dict(pamp=pamp, mode='vc', hold=-65*mV, r_input=r_input, r_access=r_access, c_soma=c_soma)
    tp, _ = create_test_pulse(**tp_kwds)
    check_analysis(tp, _['soma'], tp_kwds, only=[only])


def test_insignificant_transient():
    tp_kwds = dict(pamp=-10*mV, mode='vc', c_soma=1*pF, c_pip=0.1*pF, r_input=1*MOhm, r_access=5*MOhm)
    tp, _ = create_test_pulse(**tp_kwds)
    check_analysis(tp, _['soma'], tp_kwds)


def test_with_60Hz_noise():
    assert False  # TODO


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


def create_test_pulse(
        start=5*ms,
        pdur=10*ms,
        pamp=-10*pA,
        hold=0,
        mode='ic',
        dt=10*us,
        r_access=10*MOhm,
        r_input=100*MOhm,
        c_soma=30*pF,
        c_pip=5*pF,
        noise=5*pA
):
    soma = h.Section()
    soma.insert('pas')
    soma.cm = 1.0  # µF/cm²
    soma.L = soma.diam = (500 / np.pi) ** 0.5  # um (500 um²)
    soma.cm = soma.cm * c_soma / capacitance(soma)
    set_resistance(soma, r_input)
    # nln, cmat, gmat, y, y0, b = set_pip_cap(c_pip)

    settle = 200 * ms
    pulse = np.ones((int((settle + start + pdur + settle) // dt),)) * hold
    pulse[int((settle + start) // dt):int((settle + start + pdur) // dt)] = pamp

    def run():
        vinit = -60  # mV

        h.init()
        h.finitialize(vinit)

        h.dt = dt / ms
        h.continuerun((settle + start + pdur + settle) / ms)

    if mode == 'ic':
        pipette_halfangle = np.deg2rad(20 / 2)
        base_radius = 0.5 * 0.86 * mm
        tip_radius = 0.5 * 1.0 * um
        length = base_radius / np.tan(pipette_halfangle)

        resistivity = np.pi * base_radius * tip_radius * r_access / length

        n_pip_sections = 20
        pip_sections = []

        # make a series of connected sections to approximate a truncated conic conductor.
        # sections will have progressively smaller radius. section lengths are chosen
        # such that all sections have equal resistance
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

            # print(f"section {i}: r={cyl_r/um:.2f}um, l={l/um:.2f}um")

            # l = axial_resistance_per_section / (resistivity * cross_sectional_area)
            sec = h.Section()
            sec.nseg = 1
            sec.L = l / um
            sec.diam = 2 * cyl_r / um
            sec.Ra = resistivity / cm
            if i > 0:
                pip_sections[-1].connect(sec(0), 1)
            pip_sections.append(sec)
            next_radius -= l * np.tan(pipette_halfangle)

        pip_sections[-1].connect(soma(0.5), 1)

        total_surface_area = np.sum([section_surface(sec) for sec in pip_sections])
        pip_cap_per_area = c_pip / total_surface_area
        for sec in pip_sections:
            sec.cm = pip_cap_per_area * cm**2 / uF

        pre_ic = _make_ic_command(pip_sections[0](0), hold, 0, settle + start)
        pulse_ic = _make_ic_command(pip_sections[0](0), pamp, settle + start, pdur)
        post_ic = _make_ic_command(pip_sections[0](0), hold, settle + start + pdur, settle)

        # ic_rec = h.Vector()
        # ic_rec.record(soma(0.5)._ref_v)
        # pip_rec1 = h.Vector()
        # pip_rec1.record(pip_thick(1)._ref_v)
        pip_rec0 = h.Vector()
        pip_rec0.record(pip_sections[0](0)._ref_v)

        run()
        # out = ic_rec.as_numpy() * mV
        out = pip_rec0.as_numpy() * mV
    else:
        vc = h.SEClamp(soma(0.5))
        vc.rs = r_access / MOhm

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
    pulse = pulse[int(settle // dt):int((settle + start + pdur + settle) // dt)]
    # pg.plot(pulse, title=f'{mode} command')
    # pg.plot((pip_rec1.as_numpy() * mV)[int(settle // dt):int((settle + start + pdur + settle) // dt)], title=f'{mode} pipette(1) voltage')
    # pg.plot((pip_rec0.as_numpy() * mV)[int(settle // dt):int((settle + start + pdur + settle) // dt)], title=f'{mode} pipette(0) voltage')

    tp = PatchClampTestPulse(
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
            holding_current=hold if mode == 'ic' else None,
            holding_potential=hold if mode == 'vc' else None,
        ),
    )
    return tp, locals()  # NEURON blows up if GC deletes objects before we're done


def expected_testpulse_values(cell, tp_kwds):
    values = {
        'access_resistance': tp_kwds.get('r_access', 10*MOhm),
        'capacitance': capacitance(cell),
        'input_resistance': resistance(cell),
    }
    if tp_kwds.get('mode', 'ic') == 'ic':
        # values['baseline_potential'] = 0  # TODO
        values['baseline_current'] = tp_kwds.get('hold', 0)
    else:
        values['baseline_potential'] = tp_kwds.get('hold', 0)
        # values['baseline_current'] = 0  # TODO

    return values


def check_analysis(pulse, cell, tp_kwds, only=None):
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
    if only:
        expected = {k: v for k, v in expected.items() if k in only}
    for k, v1 in expected.items():
        v2 = measured[k]
        if v1 is None:
            print(f"Expected None for {k}, measured {v2}")
            continue
        elif v2 is None:
            print(f"Expected {v1} for {k}, measured None")
            continue
        if v1 == 0:
            err = abs(v1 - v2)
        else:
            err = abs((v1 - v2) / v1)
        print(f"Expected {v1:g} for {k}, got {v2:g} (err {err:g})" if err > err_tolerance[k] else f"Expected {v1:g} for {k}, got {v2:g}")
        if err > err_tolerance[k]:
            print(f"Expected {v1:g} for {k}, got {v2:g} (err {err:g} > {err_tolerance[k]:g})")
            mistakes = True
    assert not mistakes


if __name__ == '__main__':
    # vc_kwds = dict(pamp=-85*mV, mode='vc', hold=-65*mV, r_input=200*MOhm, r_access=15*MOhm)
    # vc_tp, vc_locals = create_test_pulse(**vc_kwds)
    # vc_tp.plot()
    # check_analysis(vc_tp, vc_locals['soma'], vc_kwds)

    ic_kwds = dict(pamp=-100*pA, mode='ic', r_input=200*MOhm, r_access=10*MOhm, pdur=50*ms, c_pip=1*pF, c_soma=100*pF)
    ic_tp, ic_locals = create_test_pulse(**ic_kwds)

    ic_tp.plot()
    pg.exec()

    check_analysis(ic_tp, ic_locals['soma'], ic_kwds)
