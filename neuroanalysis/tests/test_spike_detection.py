import glob
import os
import pickle

import numpy as np
import pytest

import neuroanalysis
from neuroanalysis.data import TSeries
from neuroanalysis.neuronsim.model_cell import ModelCell
from neuroanalysis.spike_detection import SpikeDetectTestCase, detect_evoked_spikes
from neuroanalysis.units import pA, MOhm, pF, us, ms

path = os.path.join(os.path.dirname(neuroanalysis.__file__), '..', 'test_data', 'evoked_spikes', '*.pkl')
spike_files = sorted(glob.glob(path))

test_ui = None


@pytest.mark.parametrize('test_file', spike_files)
def test_spike_detection(request, test_file):
    global test_ui
    try:
        audit = request.config.getoption('audit')
    except Exception:
        audit = False
    if audit and test_ui is None:
        from neuroanalysis.ui.spike_detection import SpikeDetectTestUi

        test_ui = SpikeDetectTestUi()

    print("test:", test_file)
    tc = SpikeDetectTestCase()
    tc.load_file(test_file)
    if audit:
        tc.audit_test(test_ui)
    else:
        tc.run_test()


@pytest.mark.xfail(reason="vc refractory keeps the first event, not the strongest", strict=True)
def test_strong_spike_is_not_masked_by_a_weaker_earlier_event():
    """A weak event must not suppress a steeper spike inside the refractory window.

    In this sweep the genuine spike is an order of magnitude steeper than the noise
    preceding it, but two noise events fall within 1 ms and used to claim the spike
    slots first, leaving the real spike discarded as "too soon after a spike".
    """
    path = os.path.join(
        os.path.dirname(neuroanalysis.__file__), '..', 'test_data', 'evoked_spikes',
        '1484952266.115_0_6_1.346.pkl')
    with open(path, 'rb') as fh:
        case = pickle.load(fh)

    spikes = detect_evoked_spikes(**case['input_args'])

    # the steepest slope present in this sweep belongs to the genuine spike
    assert min(s['max_slope'] for s in spikes) < -4e-6


def test_model_spike_detection():
    # Need to fill this function up with many more tests, especially 
    # measuring against real data.
    dt = 10*us
    start = 5*ms
    duration = 2*ms

    resp = create_test_pulse(start=5*ms, pamp=100*pA, pdur=2*ms, mode='ic', dt=dt)
    pulse_edges = resp['primary'].t0 + start, resp['primary'].t0 + start + duration
    spikes = detect_evoked_spikes(resp, pulse_edges)
    assert len(spikes) == 0
    
    resp = create_test_pulse(start=5*ms, pamp=1000*pA, pdur=2*ms, mode='ic', dt=dt)
    pulse_edges = resp['primary'].t0 + start, resp['primary'].t0 + start + duration
    spikes = detect_evoked_spikes(resp, pulse_edges)
    assert len(spikes) == 1


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

    return result


def main():
    global plt, dt, start, duration, pulse_edges, ui, test_pulse, ra, amp

    import pyqtgraph as pg
    from neuroanalysis.ui.spike_detection import SpikeDetectUI

    plt = pg.plot(labels={'left': ('Vm', 'V'), 'bottom': ('time', 's')})
    dt = 10 * us
    start = 5 * ms
    duration = 2 * ms
    pulse_edges = start, start + duration
    ui = SpikeDetectUI()

    def test_pulse(amp, ra):
        # Simulate pulse response
        resp = create_test_pulse(start=start, pamp=amp, pdur=duration, mode='ic', r_access=ra)

        # Test spike detection
        pri = resp['primary']
        pri.t0 = 0
        spikes = detect_evoked_spikes(resp, pulse_edges, ui=ui)
        print(spikes)
        pen = ['r', 'y', 'g', 'b'][len(spikes)]

        # plot in green if a spike was detected
        plt.plot(pri.time_values, pri.data, pen=pen)

    # Iterate over a series of increasing pulse amplitudes
    for ra in [10 * MOhm, 100 * MOhm]:
        for amp in np.arange(0 * pA, 1500 * pA, 100 * pA):
            print("Amp: %f   Raccess: %f" % (amp, ra))
            test_pulse(amp, ra)

            # redraw after every new test
            pg.QtWidgets.QApplication.processEvents()


if __name__ == '__main__':
    main()
