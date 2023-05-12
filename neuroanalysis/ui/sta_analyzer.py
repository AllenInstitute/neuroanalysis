import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.parametertree as pt
from .cell_selector import CellSelector
from .event_detection import EventDetector
from .triggered_average import TriggeredAverager
from ..data import TSeries


class STAAnalyzer(QtWidgets.QWidget):
    """Analyzer for running spike-triggered averaging on BOb calcium imaging data.
    
    Features:
    
    * Select cells from a list of pre-segmented ROIs
    * Detect spikes using exponential deconvolution
    * Plot calcium signal and deconvolved signal
    * Display spike-triggered stimulus average with a delay window 
    
    """
    def __init__(self, boc, expt_id, cell_id):
        self.boc = boc
        self.data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt_id)
        self.lsn_tmp = self.data_set.get_stimulus_template('locally_sparse_noise')
        
        # setup cell selector
        cells = self.data_set.get_cell_specimen_ids()
        self.cell_selector = CellSelector()
        
        roi_img = (self.data_set.get_roi_mask_array() * np.array(cells)[:,None,None]).max(axis=0)
        max_img = self.data_set.get_max_projection()
        self.cell_selector.set_images(max_img, roi_img)
        
        # setup spike detector
        self.spike_detector = EventDetector()
        self.spike_detector.params['filter', 'cutoff'] = 10
        self.spike_detector.params['deconv const'] = 700e-3
        self.spike_detector.params['threshold'] = 2

        # setup averager
        self.averager = TriggeredAverager()
        
        # make stimulus frame locations easier to look up
        self.lsn_id = None
        
        QtWidgets.QWidget.__init__(self)
        self.hs = pg.QtWidgets.QSplitter()
        self.hs.setOrientation(pg.QtCore.Qt.Horizontal)

        self.vs1 = pg.QtWidgets.QSplitter()
        self.vs1.setOrientation(pg.QtCore.Qt.Vertical)

        self.params = pt.Parameter(name='params', type='group', children=[
            self.cell_selector.params,
            self.spike_detector.params,
            self.averager.params,
        ])

        self.tree = pt.ParameterTree(showHeader=False)
        self.tree.setParameters(self.params, showTop=False)
        self.vs1.addWidget(self.tree)

        self.expt_imv = pg.ImageView()
        self.cell_selector.set_imageview(self.expt_imv)
        
        self.vs1.addWidget(self.expt_imv)

        self.vs2 = pg.QtWidgets.QSplitter()
        self.vs2.setOrientation(pg.QtCore.Qt.Vertical)

        self.plt1 = pg.PlotWidget()
        self.plt2 = pg.PlotWidget()
        self.plt2.setXLink(self.plt1)
        self.spike_detector.set_plots(self.plt1, self.plt2)

        self.sta_imv = pg.ImageView()
        self.averager.set_imageview(self.sta_imv)

        self.vs2.addWidget(self.plt1)
        self.vs2.addWidget(self.plt2)
        self.vs2.addWidget(self.sta_imv)

        self.hs.addWidget(self.vs1)
        self.hs.addWidget(self.vs2)
        
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.hs)
        
        self.resize(1400, 800)
        self.hs.setSizes([600, 600])
        self.show()
        
        self.cell_selector.cell_selection_changed.connect(self.loadCell)
        self.spike_detector.parameters_changed.connect(self.updateSpikes)
        self.averager.parameters_changed.connect(self.updateOutput)

        self.loadCell()

    def loadCell(self):
        cell_id = self.cell_selector.selected_id()
        if cell_id is None:
            return
        
        data = self.data_set.get_dff_traces([cell_id])
        
        if self.lsn_id is None:
            lsn_st = self.data_set.get_stimulus_table('locally_sparse_noise')
            self.lsn_id = np.zeros(len(data[0]), dtype='int') - 1
            for i,frame in lsn_st.iterrows():
                self.lsn_id[frame['start']:frame['end']] = frame['frame']
        
        self.updateSpikes()

    def updateSpikes(self):
        cell_id = self.cell_selector.selected_id()
        self.data = self.data_set.get_dff_traces([cell_id])
        trace = TSeries(self.data[1][0], time_values=self.data[0])
        self.events = self.spike_detector.process(trace)
        self.updateOutput()
        
    def updateOutput(self):
        t = self.data[0]
        dt = t[1] - t[0]
        sta = self.averager.process(self.events, self.lsn_tmp, self.lsn_id, dt)
