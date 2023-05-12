import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from neuroanalysis.miesnwb import MiesNwb
from ..signal import SignalBlock

from .sweep_view import SweepView
from .analyzer_view import AnalyzerView
from ...util.merge_lists import merge_lists


class MiesNwbExplorer(QtWidgets.QSplitter):
    """Widget for listing and selecting recordings in a MIES-generated NWB file.
    """
    selection_changed = QtCore.Signal(object)
    channels_changed = QtCore.Signal(object)
    check_state_changed = QtCore.Signal(object)

    def __init__(self, nwb=None):
        QtWidgets.QSplitter.__init__(self)
        self.setOrientation(QtCore.Qt.Vertical)

        self._nwb = None
        self._channel_selection = {}

        self._sel_box = QtWidgets.QWidget()
        self._sel_box_layout = QtWidgets.QHBoxLayout()
        self._sel_box_layout.setContentsMargins(0, 0, 0, 0)
        self._sel_box.setLayout(self._sel_box_layout)
        self.addWidget(self._sel_box)
        self.sweep_tree = QtWidgets.QTreeWidget()
        columns = ['ID', 'Stim Name', 'Clamp Mode', 'Holding V', 'Holding I']
        self.sweep_tree.setColumnCount(len(columns))
        self.sweep_tree.setHeaderLabels(columns)
        self.sweep_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._sel_box_layout.addWidget(self.sweep_tree)
        
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setMaximumWidth(50)
        self._sel_box_layout.addWidget(self.channel_list)
        self.channel_list.itemChanged.connect(self._channel_list_changed)

        self.meta_tree = QtWidgets.QTreeWidget()
        self.addWidget(self.meta_tree)

        self.set_nwb(nwb)

        self.sweep_tree.itemSelectionChanged.connect(self._selection_changed)
        self.sweep_tree.itemChanged.connect(self._tree_item_changed)

    def set_nwb(self, nwb):
        self._nwb = nwb
        self._channel_selection = {}
        self.update_sweep_tree()

    def update_sweep_tree(self):
        self.sweep_tree.clear()

        if self._nwb is None:
            return

        for i,sweep in enumerate(self._nwb.contents):
            recs = sweep.recordings
            stim = recs[0].stimulus
            stim_name = '' if stim is None else stim.description
            modes = ''
            V_holdings = ''
            I_holdings = ''
            for rec in sweep.recordings:
                if not hasattr(rec, 'clamp_mode'):
                    modes += "-"
                    V_holdings += "-"
                    I_holdings += "-"
                    continue
                    
                if rec.clamp_mode == 'vc':
                    modes += 'V'
                else:
                    modes += 'I'
                hp = rec.rounded_holding_potential
                if hp is not None:
                    V_holdings += '%d '% (int(hp*1000))
                else:
                    V_holdings += '?? '

                hc = rec.holding_current
                if hc is not None:
                    I_holdings += '%d '% (int(hc*1e12))
                else:
                    I_holdings += '?? '
                    
            item = QtWidgets.QTreeWidgetItem([str(i), stim_name, modes, V_holdings, I_holdings])
            item.setCheckState(0, QtCore.Qt.Unchecked)
            item.data = sweep
            self.sweep_tree.addTopLevelItem(item)
 
        self.sweep_tree.header().resizeSections(QtWidgets.QHeaderView.ResizeToContents)

    def selection(self):
        """Return a list of selected groups and/or sweeps. 
        """
        items = self.sweep_tree.selectedItems()
        selection = []
        for item in items:
            if item.parent() in items:
                continue
            selection.append(item.data)
        return selection

    def checked_items(self, _root=None):
        """Return a list of items that have been checked.
        """
        if _root is None:
            _root = self.sweep_tree.invisibleRootItem()
        checked = []
        if _root.checkState(0) == QtCore.Qt.Checked:
            checked.append(_root.data)
        for i in range(_root.childCount()):
            checked.extend(self.checked_items(_root.child(i)))
        return checked

    def selected_channels(self):
        chans = []
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                chans.append(item.channel_index)
        return chans

    def _update_channel_list(self):
        self.channel_list.itemChanged.disconnect(self._channel_list_changed)
        try:
            # clear channel list
            while self.channel_list.count() > 0:
                self.channel_list.takeItem(0)
            
            # bail out if nothing is selected
            sel = self.selection()
            if len(sel) == 0:
                return
            
            # get a list of all channels across all selected items
            channels = []
            for item in sel:
                #if isinstance(item, SweepGroup):
                    #if len(item.sweeps) == 0:
                        #continue
                    #item = item.sweeps[0]
                channels.extend(item.devices)
            channels = list(set(channels))
            channels.sort()
            
            # add new items to the channel list, all selected
            for ch in channels:
                item = QtWidgets.QListWidgetItem(str(ch))
                item.channel_index = ch
                self._channel_selection.setdefault(ch, True)
                # restore previous check state, if any.
                checkstate = QtCore.Qt.Checked if self._channel_selection.setdefault(ch, True) else QtCore.Qt.Unchecked
                item.setCheckState(checkstate)
                self.channel_list.addItem(item)
        finally:
            self.channel_list.itemChanged.connect(self._channel_list_changed)

    def _selection_changed(self):
        sel = self.selection()
        if len(sel) == 1:
            sweep = sel[0]
            self.meta_tree.setColumnCount(len(sweep.devices)+1)
            self.meta_tree.setHeaderLabels([""] + [str(dev) for dev in sweep.devices])
            self.meta_tree.clear()
            self._populate_meta_tree([dev.all_meta for dev in sweep.recordings], self.meta_tree.invisibleRootItem())
            for i in range(self.meta_tree.columnCount()):
                self.meta_tree.resizeColumnToContents(i)
        else:
            self.meta_tree.clear()
        self._update_channel_list()
        self.selection_changed.emit(sel)

    def _populate_meta_tree(self, meta, root):
        keys = list(meta[0].keys())
        for m in meta[1:]:
            keys = merge_lists(keys, list(m.keys()))
        
        for k in keys:
            vals = [m.get(k) for m in meta]
            if isinstance(vals[0], dict):
                item = QtWidgets.QTreeWidgetItem([k] + [''] * len(meta))
                self._populate_meta_tree(vals, item)
            else:
                item = QtWidgets.QTreeWidgetItem([k] + [str(v) for v in vals])
            root.addChild(item)

    def _tree_item_changed(self, item, col):
        if col != 0:
            return
        self.check_state_changed.emit(self)
        
    def _channel_list_changed(self, item):
        self.channels_changed.emit(self.selected_channels())
        self._channel_selection[item.channel_index] = item.checkState() == QtCore.Qt.Checked


class MiesNwbViewer(QtWidgets.QWidget):
    """Combination of a MiesNwvExplorer for selecting sweeps and a tab widget
    containing multiple views, each performing a different analysis.
    """
    analyzer_changed = QtCore.Signal(object)
    
    def __init__(self, nwb=None):
        QtWidgets.QWidget.__init__(self)
        self.nwb = nwb
        
        self.layout = QtWidgets.QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.hsplit = QtWidgets.QSplitter()
        self.hsplit.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.hsplit, 0, 0)
        
        self.vsplit = QtWidgets.QSplitter()
        self.vsplit.setOrientation(QtCore.Qt.Vertical)
        self.hsplit.addWidget(self.vsplit)

        self.explorer = MiesNwbExplorer(self.nwb)
        self.explorer.selection_changed.connect(self.data_selection_changed)
        self.explorer.channels_changed.connect(self.data_selection_changed)
        self.vsplit.addWidget(self.explorer)

        self.ptree = pg.parametertree.ParameterTree()
        self.vsplit.addWidget(self.ptree)
        
        self.tabs = QtWidgets.QTabWidget()
        self.hsplit.addWidget(self.tabs)
        
        self.reload_btn = QtWidgets.QPushButton("Reload views")
        self.reload_btn.clicked.connect(self.reload_views)
        self.vsplit.addWidget(self.reload_btn)

        self.views = []
        self.create_views()
        
        self.resize(1400, 800)
        self.hsplit.setSizes([600, 800])

        self.tab_changed()
        self.tabs.currentChanged.connect(self.tab_changed)

    def set_nwb(self, nwb):
        self.nwb = nwb
        self.explorer.set_nwb(nwb)

    def data_selection_changed(self):
        sweeps = self.selected_sweeps()
        chans = self.selected_channels()
        with pg.BusyCursor():
            self.tabs.currentWidget().data_selected(sweeps, chans)

    def tab_changed(self):
        w = self.tabs.currentWidget()
        if w is None:
            self.ptree.clear()
            return
        self.ptree.setParameters(w.params, showTop=False)
        sweeps = self.selected_sweeps()
        chans = self.selected_channels()
        w.data_selected(sweeps, chans)
        self.analyzer_changed.emit(self)
        
    def selected_analyzer(self):
        return self.tabs.currentWidget()

    def selected_sweeps(self, selection=None):
        if selection is None:
            selection = self.explorer.selection()
        sweeps = []
        for item in selection:
            #if isinstance(item, SweepGroup):
                #sweeps.extend(item.sweeps)
            #else:
            sweeps.append(item)
        return sweeps
    
    def checked_sweeps(self):
        selection = self.explorer.checked_items()
        sweeps = []
        for item in selection:
            #if isinstance(item, SweepGroup):
                #sweeps.extend(item.sweeps)
            #else:
            sweeps.append(item)
        return sweeps
    
    def selected_channels(self):
        return self.explorer.selected_channels()

    def reload_views(self):
        """Remove all existing views, reload their source code, and create new
        views.
        """
        self.clear_views()
        pg.reload.reloadAll(debug=True)
        self.create_views()

    def clear_views(self):
        with SignalBlock(self.tabs.currentChanged, self.tab_changed):
            self.tabs.clear()
            self.views = []
        
    def create_views(self):
        self.clear_views()
        self.views = [
            ('Sweep', SweepView(self)),
            ('Sandbox', AnalyzerView(self)),
        ]

        for name, view in self.views:
            self.tabs.addTab(view, name)



class AnalysisView(QtWidgets.QWidget):
    """Example skeleton for an analysis view.
    """
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        
        # Views must have self.params
        # This implements the controls that are unique to this view.
        self.params = pg.parametertree.Parameter(name='params', type='group', children=[
            {'name': 'lowpass', 'type': 'float', 'value': 0, 'limits': [0, None], 'step': 1},
            {'name': 'average', 'type': 'bool', 'value': False},
        ])
        self.params.sigTreeStateChanged.connect(self._update_analysis)

    def data_selected(self, sweeps, channels):
        """Called when the user selects a different set of sweeps.
        """
        self.sweeps = sweeps
        self.channels = channels
        self.update_analysis()

    def update_analysis(self, param, changes):
        """Called when the user changes control parameters.
        """
        pass


        
if __name__ == '__main__':
    import sys
    from pprint import pprint
    pg.dbg()
    
    filename = sys.argv[1]
    nwb = MiesNwb(filename)
    # sweeps = nwb.sweeps()
    # traces = sweeps[0].traces()
    # # pprint(traces[0].meta())
    # groups = nwb.sweep_groups()
    # for i,g in enumerate(groups):
    #     print "--------", i, g
    #     print g.describe()

    # d = groups[7].data()
    # print d.shape

    app = pg.mkQApp()
    w = MiesNwbViewer(nwb)
    w.show()
    # w.show_group(7)
