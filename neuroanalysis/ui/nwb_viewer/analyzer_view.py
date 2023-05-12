import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from ..plot_grid import PlotGrid


class AnalyzerView(QtWidgets.QWidget):
    """A sweep analyzer of unspecified function.
    """
    def __init__(self, parent=None):
        self.sweeps = []

        QtWidgets.QWidget.__init__(self, parent)

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plots = PlotGrid(parent=self)
        self.layout.addWidget(self.plots, 0, 0)

        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            # Spin boxes
            #{'name': 'value1', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 1},
            #{'name': 'value2', 'type': 'float', 'value': 10e-3, suffix='V', dec=True, minStep=1e-6},
            
            # Check box
            #{'name': 'average', 'type': 'bool', 'value': False},
            
            # Dropdown list
            #{'name': 'average', 'type': 'list', 'values': ["option 1", "option 2", "option 3"]},
            
        ])
        self.params.sigTreeStateChanged.connect(self._update_plots)

    def data_selected(self, sweeps, channels):
        self.sweeps = sweeps
        self.channels = channels
        self._update_plots()

    def _update_plots(self):
        sweeps = self.sweeps
        channels = self.channels
        
        # clear all plots
        self.plots.clear()
        
        # If there are no selected sweeps or channels, return now
        if len(sweeps) == 0 or len(channels) == 0:
            return
        
        # Resize the plot grid based on the number of selected channels
        n_channels = len(channels)
        self.plots.set_shape(n_channels, 1)
        
        # Set some plot options to improve performance
        self.plots.setClipToView(True)
        self.plots.setDownsampling(True, True, 'peak')

        # Link all x axes
        self.plots.setXLink(self.plots[0, 0])
        
        # Iterate over selected channels of all sweeps, plotting traces one at a time
        for sweep in sweeps:
            for i in range(n_channels):
                chan = channels[i]
                trace = sweep[chan]
                self.plots[i,0].plot(trace['primary'].time_values, trace['primary'].data, antialias=True)
                
        # label plots
        for i,ch in enumerate(channels):
            self.plots[i,0].setLabels(left="Channel %d" % ch)
