import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.parametertree as pt
from scipy.ndimage import gaussian_filter

from ..filter import remove_artifacts, bessel_filter


class SignalFilter(QtCore.QObject):
    """A user-configurable signal filter.
    
    todo: turn this into a flowchart-based filter
    """
    def __init__(self):
        self.params = pt.Parameter.create(name='filter', type='bool', value=True, children=[
            #{'name': 'filter type', 'type': 'list', 'values': ['bessel', 'butterworth']},
            {'name': 'cutoff', 'type': 'float', 'value': 1000, 'step': 0.5, 'minStep': 1e-3, 'dec': True, 'limits': [0, None], 'suffix': 'Hz', 'siPrefix': True},
            {'name': 'order', 'type': 'int', 'value': 4, 'step': 1, 'limits': [1, None]},
            {'name': 'passband', 'type': 'list', 'values': ['low', 'high']},
        ])
        
    def process(self, trace):
        if self.params.value() is False:
            return trace
        return bessel_filter(trace, cutoff=self.params['cutoff'], order=self.params['order'], btype=self.params['passband'])
        

class ArtifactRemover(QtCore.QObject):
    """
    """
    def __init__(self, user_width=False):
        self.params = pt.Parameter.create(name='remove artifacts', type='bool', value=True, children=[
            {'name': 'width', 'type': 'float', 'value': 200e-6, 'step': 1e-5, 'limits': [0, None], 'suffix': 's', 'siPrefix': True, 'visible': user_width},
            {'name': 'fill window', 'type': 'float', 'value': 100e-6, 'step': 1e-5, 'limits': [0, None], 'suffix': 's', 'siPrefix': True},
        ])
        self._user_width = user_width
        
    def process(self, trace, edges):
        if self.params.value() is False:
            return trace
        if self._user_width:
            w = self.params['width'] / trace.dt
            edges = [(t, t+w) for t in edges]
        return remove_artifacts(trace, edges, self.params['fill window'])
        
