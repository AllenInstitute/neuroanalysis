import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.parametertree as pt
import numpy as np
import scipy.ndimage as ndi

from ..event_detection import threshold_events, exp_deconvolve
from ..baseline import float_mode
from .filter import SignalFilter


class EventDetector(QtCore.QObject):
    """Analyzer to detect transient events from signals such as calcium indicator
    and current clamp recordings.
    
    The basic algorithm is:
    
    1. Lowpass input signal using gaussian filter
    2. Exponential deconvolution to isolate spikes
       (Richardson & Silberberg, J. Neurophysiol 2008)
    3. Threshold detection of events


    Signals
    -------
    parameters_changed(self):
        Emitted whenever a parameter has changed that would affect the output
        of the analyzer.
        
    """
    parameters_changed = QtCore.Signal(object)  # self
    
    def __init__(self):
        QtCore.QObject.__init__(self)
        
        self.filter = SignalFilter()
        
        self.params = pt.Parameter(name='Spike detection', type='group', children=[
            {'name': 'deconv const', 'type': 'float', 'value': 0.01, 'suffix': 's', 'siPrefix': True, 'dec': True, 'minStep': 1e-4},
            {'name': 'threshold', 'type': 'float', 'value': 0.05, 'dec': True, 'minStep': 1e-12},
            self.filter.params,
        ])
        self.sig_plot = None
        self.deconv_plot = None
        
        self.sig_trace = None
        self.vticks = None
        self.deconv_trace = None
        self.threshold_line = None
        
        self.params.sigTreeStateChanged.connect(self._parameters_changed)
        self.params.child('threshold').sigValueChanged.connect(self._threshold_param_changed)
        
    def set_plots(self, plt1=None, plt2=None):
        """Connect this detector to two PlotWidgets where data should be displayed.
        
        The first plot will contain the lowpass-filtered trace and tick marks
        for detected events. The second plot will contain the deconvolved signal
        and a draggable threshold line.
        """
        self.sig_plot = plt1
        if plt1 is not None:
            if self.sig_trace is None:
                self.sig_trace = pg.PlotDataItem()
                self.vticks = pg.VTickGroup(yrange=[0.0, 0.05])
            plt1.addItem(self.sig_trace)
            plt1.addItem(self.vticks)
        
        self.deconv_plot = plt2
        if plt2 is not None:
            if self.deconv_trace is None:
                self.deconv_trace = pg.PlotDataItem()
                self.threshold_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
                self.threshold_line.setValue(self.params['threshold'])
                self.threshold_line.sigPositionChanged.connect(self._threshold_line_moved)
            plt2.addItem(self.deconv_trace)
            plt2.addItem(self.threshold_line)
        
    def process(self, trace, show=True):
        """Return a table (numpy record array) of events detected in a time series.
        
        Parameters
        ----------
        trace : data.TSeries instance
            Signal values to process for events (for example, a single calcium
            signal trace or a single electrode recording).
        show : bool
            If True, then processed data will be displayed in the connected
            plots (see `set_plots()`).
        
        Returns
        -------
        events : numpy record array
            The returned table has several fields:
            
            * index: the index in *data* at which an event began
            * len: the length of the deconvolved event in samples
            * sum: the integral of *data* under the deconvolved event curve
            * peak: the peak value of the deconvolved event
        """
        y = trace.data
        dt = trace.dt
        
        
        # Exponential deconvolution; see Richardson & Silberberg, J. Neurophysiol 2008
        tau = self.params['deconv const']
        diff = exp_deconvolve(trace, tau)
        
        # remove baseline
        bsub = diff.data - float_mode(diff.data, bins=200)
        bsub = diff.copy(data=bsub)
        
        # filter
        filt = self.filter.process(bsub)        
        
        self.events = threshold_events(filt.data, self.threshold_line.value())
        #self.events = self.events[self.events['sum'] > 0]

        if show:
            if self.sig_plot is not None:
                t = trace.time_values
                self.sig_trace.setData(t, y)
                self.vticks.setXVals(t[self.events['index']])
                self.vticks.update()  # this should not be needed..
            if self.deconv_plot is not None:
                self.deconv_trace.setData(filt.time_values, filt.data)

        return self.events
        
    def _parameters_changed(self):
        self.parameters_changed.emit(self)
    
    def _threshold_line_moved(self):
        # link line position to threshold parameter
        self.params.child('threshold').setValue(self.threshold_line.value(), blockSignal=self._threshold_param_changed)
        
    def _threshold_param_changed(self):
        # link line position to threshold parameter
        if self.threshold_line is not None:
            self.threshold_line.setValue(self.params['threshold'])
    

