import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.parametertree as pt

from ..baseline import mode_detrend


class BaselineRemover(QtCore.QObject):
    """Removes baseline from signal.
    
    todo: turn this into a flowchart-based filter
    """
    def __init__(self):
        self.params = pt.Parameter.create(name='baseline', type='bool', value=True, children=[
            {'name': 'window', 'type': 'float', 'value': 10e-3, 'dec': True, 'minStep': 1e-6, 'limits': [1e-6, None], 'suffix': 's', 'siPrefix': True},
        ])
        
    def process(self, trace):
        if self.params.value() is False:
            return trace
        w = int(self.params['window'] / trace.dt)
        data = mode_detrend(trace.data, window=w)
        return trace.copy(data=data)
        

