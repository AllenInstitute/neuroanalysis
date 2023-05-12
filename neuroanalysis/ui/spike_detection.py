from __future__ import division, print_function

import time
import pyqtgraph as pg
import pyqtgraph.console
from .user_test import UserTestUi


class SpikeDetectUI(object):
    """Used to display details of spike detection analysis.
    """
    def __init__(self, title=None):
        self.pw = pg.GraphicsLayoutWidget()
        self.plt1 = self.pw.addPlot(title=title)
        self.plt2 = self.pw.addPlot(row=1, col=0)
        self.plt2.setXLink(self.plt1)
        self.plt3 = self.pw.addPlot(row=2, col=0)
        self.plt3.setXLink(self.plt1)
        
        self.console = pg.console.ConsoleWidget()
        
        self.widget = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Vertical)
        self.widget.addWidget(self.pw)        
        self.widget.addWidget(self.console)
        self.widget.resize(1000, 900)
        self.widget.setSizes([800, 200])
        self.widget.show()
    
    def clear(self):
        self.plt1.clear()
        self.plt2.clear()
        self.plt3.clear()

    def show_result(self, spikes):
        for plt in [self.plt1, self.plt2, self.plt3]:
            if spikes is None:
                continue
            for spike in spikes:
                if spike.get('onset_time') is not None:
                    plt.addLine(x=spike['onset_time'])
                if spike.get('max_slope_time') is not None:
                    plt.addLine(x=spike['max_slope_time'], pen='b')
                if spike.get('peak_time') is not None:
                    plt.addLine(x=spike['peak_time'], pen='g')


class SpikeDetectTestUi(UserTestUi):
    """UI for manually pass/failing spike detection unit tests.
    """
    def __init__(self):
        expected_display = SpikeDetectUI('expected result')
        current_display = SpikeDetectUI('current result')
        UserTestUi.__init__(self, expected_display, current_display)
