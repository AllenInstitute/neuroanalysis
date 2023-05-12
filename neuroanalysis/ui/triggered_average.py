import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.parametertree as pt
import numpy as np
import scipy.ndimage as ndi


class TriggeredAverager(QtCore.QObject):
    parameters_changed = QtCore.Signal(object)  # self

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.params = pt.Parameter(name='Triggered Average', type='group', children=[
            {'name': 'on/off', 'type': 'list', 'values': ['any', 'on', 'off']},
            {'name': 'delay', 'type': 'float', 'value': -0.2, 'suffix': 's', 'siPrefix': True, 'step': 50e-3},
            {'name': 'delay range', 'type': 'float', 'value': 1.0, 'limits': [0,None], 'suffix': 's', 'siPrefix': True, 'step': 50e-3},
            {'name': 'blur STA', 'type': 'float', 'value': 1.0, 'limits': [0,None], 'step': 0.5},
        ])
        self.imgview = None
        self.params.sigTreeStateChanged.connect(self.parameters_changed)
        
    def set_imageview(self, imv):
        self.imgview = imv
        
    def process(self, events, stimuli, stim_index, dt, show=True):
        inds = events['index'] - int(self.params['delay'] / dt)

        if self.params['on/off'] == 'on':
            stimuli = np.clip(stimuli, 127, 255)
        elif self.params['on/off'] == 'off':
            stimuli = np.clip(stimuli, 0, 127)
        
        dr = self.params['delay range']
        blur = self.params['blur STA']
        nframes = int(dr / dt)
        if nframes < 2:
            frames = stim_index[inds]
            mask = frames > 0
            stimuli = stimuli[frames[mask]]
            sta = (stimuli * events['sum'][mask][:,None,None]).mean(axis=0)
            if blur > 0:
                sta = ndi.gaussian_filter(sta, blur)
        else:
            offset = nframes // 2
            sta = np.empty((nframes,) + stimuli.shape[1:], float)
            for i in range(nframes):
                shift_inds = inds - offset + i
                mask = (shift_inds > 0) & (shift_inds < stim_index.shape[0])
                frames = stim_index[shift_inds[mask]]
                mask = frames > 0
                sta[i] = (stimuli[frames[mask]] * events['sum'][mask][:,None,None]).mean(axis=0)
            sta /= sta.mean(axis=1).mean(axis=1)[:,None,None]
            if blur > 0:
                sta = ndi.gaussian_filter(sta, (0, blur, blur))
        
        if self.imgview is not None:
            self.imgview.setImage(sta.transpose(0, 2, 1), xvals=np.arange(-offset, -offset+nframes) * dt)
            self.imgview.setCurrentIndex(sta.shape[0]/2)
        return sta


