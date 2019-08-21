import numpy as np
from neuroanalysis.data import Recording, TSeries
from neuroanalysis.test_pulse import PatchClampTestPulse
from neuroanalysis.neuronsim.model_cell import ModelCell
from neuroanalysis.units import pA, mV, MOhm, pF, us, ms


def test_test_pulse():
    # Just test against a simple R/C circuit attached to a pipette
    model_cell.enable_mechs(['leak'])
    model_cell.recording_noise = False
    
    tp = create_test_pulse(pamp=-10*pA, mode='ic', r_access=100*MOhm)    
    check_analysis(tp, model_cell)
    
    
    

model_cell = ModelCell()

    
def create_test_pulse(start=5*ms, pdur=10*ms, pamp=-10*pA, mode='ic', dt=10*us, r_access=10*MOhm, c_soma=5*pF, noise=5*pA):
    # update patch pipette access resistance
    model_cell.clamp.ra = r_access
    
    # update noise amplitude
    model_cell.mechs['noise'].stdev = noise
    
    # make pulse array
    duration = start + pdur * 3
    pulse = np.zeros(int(duration / dt))
    pstart = int(start / dt)
    pstop = pstart + int(pdur / dt)
    pulse[pstart:pstop] = pamp
    
    # simulate response
    result = model_cell.test(TSeries(pulse, dt), mode)
    
    # generate a PatchClampTestPulse to test against
    tp = PatchClampTestPulse(result)

    return tp


def expected_testpulse_values(cell):
    if cell.clamp.mode == 'ic':
        values = {
            'baseline_potential': model_cell.resting_potential(),
            'baseline_current': model_cell.clamp.holding['ic'],
            'access_resistance': model_cell.clamp.ra,
            'capacitance': model_cell.soma.cap,
        }
    else:
        values = {
            'baseline_potential': model_cell.clamp.holding['vc'],
            'baseline_current': model_cell.resting_current(),
            'access_resistance': model_cell.clamp.ra,
            'capacitance': None,
        }
    values['input_resistance'] = model_cell.input_resistance()

    return values


def check_analysis(tp, cell):
    measured = tp.analysis
    expected = expected_testpulse_values(cell)
    
    # how much error should we tolerate for each parameter?
    err_tolerance = {
        'baseline_potential': 0.01,
        'baseline_current': 0.01,
        'access_resistance': 0.3,
        'input_resistance': 0.1,
        'capacitance': 0.3,
    }
    
    for k,v1 in expected.items():
        v2 = measured[k]
        if v1 is None:
            assert v2 is None
            continue
        err = abs((v1 - v2) / v2)
        if err > err_tolerance[k]:
            raise ValueError("Test pulse metric out of range: %s = %g != %g"
                "  (err %g > %g)" % (k, v2, v1, err, err_tolerance[k]))


if __name__ == '__main__':
    import pyqtgraph as pg
    
    # Just test against a simple R/C circuit attached to a pipette
    model_cell.enable_mechs(['leak'])
    model_cell.recording_noise = False
    
    tp = create_test_pulse(pamp=-10*pA, mode='ic', r_access=100*MOhm)
    tp.plot()
    
    check_analysis(tp, model_cell)
    
    print("Vm %g mV    Rm %g MOhm" % (model_cell.resting_potential()*1000, model_cell.input_resistance()/1e6))

    # Have to test VC with very low access resistance
    tp = create_test_pulse(pamp=-10*mV, mode='vc', r_access=15*MOhm)
    tp.plot()
    
    check_analysis(tp, model_cell)
    

    