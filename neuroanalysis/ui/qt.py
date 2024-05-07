from pyqtgraph.Qt import QtGui, QtWidgets, QtCore

# copy all of the Qt classes into the local namespace to work around version differences in Qt bindings
for mod in [QtGui, QtWidgets, QtCore]:
    for k,v in mod.__dict__.items():
        locals()[k] = v
