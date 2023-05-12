import os, sys
from pyqtgraph.Qt import QtWidgets
from neuroanalysis.ui.nwb_viewer import MiesNwbViewer
from neuroanalysis.miesnwb import MiesNwb

app = QtWidgets.QApplication([])

# create NWB viewer
v = MiesNwbViewer()
v.show()
v.setWindowTitle('NWB Viewer')

nwb = None
def load_nwb(filename):
    global nwb
    nwb = MiesNwb(filename)
    v.set_nwb(nwb)

# load file or set base directory from argv
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if not os.path.exists(arg):
        print("Could not find %s" % arg)
        sys.exit(-1)
    else:
        load_nwb(arg)

# start Qt event loop if this is not an interactive python session
if sys.flags.interactive == 0:
    app.exec_()
