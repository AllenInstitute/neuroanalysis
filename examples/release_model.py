from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
from neuroanalysis.synaptic_release import ReleaseModel

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

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
plots = []
for i in range(len(test_amps)):
    plots.append(win.addPlot(i, 0))
    plots[-1].setXLink(plots[0])
win.show()
plots[0].setXRange(0, 700)

# collect into list of (tvals, yvals) pairs
spike_sets = []
for i, induction in enumerate(test_amps):
    freq, y = induction
    dt = 1000 / float(freq)
    x = np.arange(len(y)) * dt
    spike_sets.append((x, y))
    plots[i].plot(x, y, pen=None, symbol='o')


# initialize the model with all gating mechanisms disabled
model = ReleaseModel()
dynamics_types = ['Dep', 'Fac', 'UR', 'SMR', 'DSR']
model.Dynamics = {k:0 for k in dynamics_types}

print("Initial parameters:", model.dict_params)
print("Bounds:", model.dict_bounds)


# Fit the model 5 times. Each time we enable another gating mechanism.
fit_params = []
for k in dynamics_types:
    model.Dynamics[k] = 1
    fit_params.append(model.run_fit(spike_sets))
    
print("----- fit complete -----")

for j,params in enumerate(fit_params):
    print(params)
    for i,spikes in enumerate(spike_sets):
        x, y = spikes
        output = model.eval(x, params.values())
        y = output[:,1]
        x = output[:,0]
        plots[i].plot(x, y, pen=(j,6))
