from collections import OrderedDict

import numpy as np

from neuroanalysis.synaptic_release import ReleaseModel


def test_release_model():
    # Sample data from an experiment probing short-term depression in synapses between
    # parvalbumin interneurons. The presynaptic cell is stimulated to fire trains of
    # 8 action potentials at 10, 20, 50, and 100Hz. The resulting IPSP amplitudes are
    # listed here:   (induction freq, [amp1, amp2, ...])
    test_amps = [
        (10,  [1.0, 0.688474, 0.541316, 0.478579, 0.447263, 0.456316, 0.451211, 0.475421]),
        (20,  [1.0, 0.674338, 0.5371  , 0.481294, 0.434435, 0.429889, 0.442057, 0.432158]),
        (50,  [1.0, 0.594617, 0.473965, 0.392039, 0.33294 , 0.325098, 0.318139, 0.299867]),
        (100, [1.0, 0.542218, 0.343102, 0.26175 , 0.222628, 0.225812, 0.201521, 0.164937]),
    ]

    # collect into list of (tvals, yvals) pairs
    spike_sets = []
    for i, induction in enumerate(test_amps):
        freq, y = induction
        dt = 1000 / float(freq)
        x = np.arange(len(y)) * dt
        spike_sets.append((x, y))

    # initialize the model with all gating mechanisms disabled
    model = ReleaseModel()
    dynamics_types = ['Dep', 'Fac', 'UR', 'SMR', 'DSR']
    model.Dynamics = {k:0 for k in dynamics_types}

    # Fit the model 5 times. Each time we enable another gating mechanism.
    fit_params = []
    for k in dynamics_types:
        model.Dynamics[k] = 1
        fit_params.append(model.run_fit(spike_sets))
        
    # Check that output matches expected values:
    expected_output = [
        OrderedDict([('a_n', 0.33925939755817169), ('Tau_r0', 1500.0), ('a_FDR', 0.10000000000000001), ('Tau_FDR', 100.0), ('a_f', 0.01), ('Tau_f', 100.0), ('p0bar', 1.0), ('a_i', -0.10000000000000001), ('Tau_i', 3000.0), ('a_D', 0.20000000000000001), ('Tau_D', 15.0), ('Tau_r', 300.00015316640139)]),
        OrderedDict([('a_n', 0.33925939755817169), ('Tau_r0', 1500.0), ('a_FDR', 0.10000000000000001), ('Tau_FDR', 100.0), ('a_f', 0.01), ('Tau_f', 100.0), ('p0bar', 1.0), ('a_i', -0.10000000000000001), ('Tau_i', 3000.0), ('a_D', 0.20000000000000001), ('Tau_D', 15.0), ('Tau_r', 300.00015316640139)]),
        OrderedDict([('a_n', 0.3952036843371185), ('Tau_r0', 1499.9974562059651), ('a_FDR', 0.89720278553956756), ('Tau_FDR', 100.00769978438043), ('a_f', 0.01), ('Tau_f', 100.0), ('p0bar', 1.0), ('a_i', -0.10000000000000001), ('Tau_i', 3000.0), ('a_D', 0.20000000000000001), ('Tau_D', 15.0), ('Tau_r', 300.0)]),
        OrderedDict([('a_n', 0.42045853418471579), ('Tau_r0', 1499.9996279831175), ('a_FDR', 0.8055063977377751), ('Tau_FDR', 100.00256876893904), ('a_f', 0.10600816095326357), ('Tau_f', 100.00205102400378), ('p0bar', 0.93605541525068015), ('a_i', -0.222958527829556), ('Tau_i', 3000.0000235761786), ('a_D', 0.20000000000000001), ('Tau_D', 15.0), ('Tau_r', 300.0)]),
        OrderedDict([('a_n', 0.12715933067171756), ('Tau_r0', 1499.9906295209564), ('a_FDR', 0.63566624213388767), ('Tau_FDR', 100.95525947989549), ('a_f', -2.8872707862021671e-18), ('Tau_f', 100.22239809493904), ('p0bar', 0.58809385996653174), ('a_i', -0.020551666671724079), ('Tau_i', 3000.0060915968161), ('a_D', 0.42977834753555072), ('Tau_D', 35.53568490867405), ('Tau_r', 300.0)]),
    ]

    test_pass = True
    for i,params in enumerate(fit_params):
        for k in params:
            if abs(params[k] - expected_output[i][k]) > max(1e-12, abs(expected_output[i][k] / 1e3)):
                test_pass = False
                print("Parameter mismatch: %d %s\t%g\t%g\t%g" % (i, k, params[k], expected_output[i][k], params[k] - expected_output[i][k]))

    assert test_pass
