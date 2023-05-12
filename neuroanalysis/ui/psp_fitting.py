from __future__ import division, print_function

import time
import numpy as np
import pyqtgraph as pg
import pyqtgraph.console
from ..fitting.psp import StackedPsp
from .user_test import UserTestUi


class PspFitUI(object):
    """Used to display details of PSP/PSC fitting analysis.
    """
    def __init__(self, title=None):
        self.pw = pg.GraphicsLayoutWidget()
        self.plt1 = self.pw.addPlot(title=title)
        
        self.console = pg.console.ConsoleWidget()
        
        self.widget = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Vertical)
        self.widget.addWidget(self.pw)        
        self.widget.addWidget(self.console)
        self.widget.resize(1000, 600)
        self.widget.setSizes([400, 200])
        self.widget.show()
    
    def clear(self):
        self.plt1.clear()

    def show_result(self, fit):
        if fit is None:
            return
        if not isinstance(fit, dict):
            fit = fit.best_values
        psp = StackedPsp()
        x = np.linspace(fit['xoffset']-10e-3, fit['xoffset']+30e-3, 5000)
        y = psp.eval(x=x, **fit)
        self.plt1.plot(x, y, pen='g')
        


class PspFitTestUi(UserTestUi):
    """UI for manually pass/failing PSP fitting unit tests.
    """
    def __init__(self):
        expected_display = PspFitUI('expected result')
        current_display = PspFitUI('current result')
        UserTestUi.__init__(self, expected_display, current_display)
        expected_display.plt1.setXLink(current_display.plt1)
        expected_display.plt1.setYLink(current_display.plt1)
