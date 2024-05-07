import pyqtgraph as pg
from . import qt


class PlotGrid(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        
        self.rows = 0
        self.cols = 0
        self.plots = []

        self.layout = qt.QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.grid = pg.GraphicsLayoutWidget()
        self.grid.ci.layout.setSpacing(0)
        self.layout.addWidget(self.grid)
        
    def __getitem__(self, item):
        return self.plots[item[0]][item[1]]
        
    def set_shape(self, rows, cols):
        assert rows * cols < 400
        if rows == self.rows and cols == self.cols:
            return
        self.remove_plots()
        
        for i in range(rows):
            row = []
            for j in range(cols):
                p = self.grid.addPlot(i, j)
                row.append(p)
            self.plots.append(row)

        self.rows = rows
        self.cols = cols
        
    @property
    def shape(self):
        return (self.rows, self.cols)
        
    def remove_plots(self):
        for i in range(self.rows):
            for j in range(self.cols):
                p = self[i, j]
                p.hide()
                p.close()
                self.grid.removeItem(p)
        self.plots = []

    def _call_on_plots(self, m, *args, **kwds):
        for row in self.plots:
            for plt in row:
                getattr(plt, m)(*args, **kwds)

    def scene(self):
        return self.grid.scene()

    def item_index(self, item):
        """Return the (row, col) that contains this item.
        """
        plts = {}
        for i,row in enumerate(self.plots):
            for j,p in enumerate(row):
                plts[p] = (i, j)
        while True:
            if item in plts:
                return plts[item]
            item = item.parentItem()
            if item is None:
                return None

    # wrap a few methods from PlotItem:
    def clear(self):
        self._call_on_plots('clear')

    def setClipToView(self, clip):
        self._call_on_plots('setClipToView', clip)

    def setDownsampling(self, *args, **kwds):
        self._call_on_plots('setDownsampling', *args, **kwds)

    def enableAutoRange(self, *args, **kwds):
        self._call_on_plots('enableAutoRange', *args, **kwds)

    def setXLink(self, *args, **kwds):
        self._call_on_plots('setXLink', *args, **kwds)

    def setYLink(self, *args, **kwds):
        self._call_on_plots('setYLink', *args, **kwds)

    def setXRange(self, *args, **kwds):
        self._call_on_plots('setXRange', *args, **kwds)

    def setYRange(self, *args, **kwds):
        self._call_on_plots('setYRange', *args, **kwds)
        