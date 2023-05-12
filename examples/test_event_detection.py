import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
from neuroanalysis.data import TSeries
from neuroanalysis.ui.event_detection import EventDetector
from neuroanalysis.ui.plot_grid import PlotGrid


pg.mkQApp()

data = np.load("test_data/synaptic_events/events1.npz")
trace_names = sorted([x for x in data.keys() if x.startswith('trace')])
traces = {n:TSeries(data[n], dt=1.0/data['sample_rates'][i]) for i,n in enumerate(trace_names)}

evd = EventDetector()
evd.params['threshold'] = 5e-10

hs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
pt = pg.parametertree.ParameterTree(showHeader=False)

params = pg.parametertree.Parameter.create(name='params', type='group', children=[
    dict(name='data', type='list', values=trace_names),
    evd.params,
])

pt.setParameters(params, showTop=False)
hs.addWidget(pt)

plots = PlotGrid()
plots.set_shape(2, 1)
plots.setXLink(plots[0, 0])
hs.addWidget(plots)

evd.set_plots(plots[0,0], plots[1,0])

def update(auto_range=False):
    evd.process(traces[params['data']])
    if auto_range:
        plots[0,0].autoRange()

evd.parameters_changed.connect(lambda: update(auto_range=False))
params.child('data').sigValueChanged.connect(lambda: update(auto_range=True))

update(auto_range=True)

hs.show()
