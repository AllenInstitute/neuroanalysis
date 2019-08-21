import sys
import json
import numpy as np
import pyqtgraph as pg
from collections import OrderedDict
from neuroanalysis.fitting import Psp
from neuroanalysis.ui.fitting import FitExplorer


pg.mkQApp()
pg.dbg()

# Load PSP data from the test_data repository
if len(sys.argv) == 1:
    data_file = 'test_data/test_psp_fit/1485904693.10_8_2NOTstacked.json'
else:
    data_file = sys.argv[1]

data = json.load(open(data_file))

y = np.array(data['input']['data'])
x = np.arange(len(y)) * data['input']['dt']


psp = Psp()
params = OrderedDict([
    ('xoffset', (10e-3, 10e-3, 15e-3)),
    ('yoffset', 0),
    ('amp', 0.1e-3),
    ('rise_time', (2e-3, 500e-6, 10e-3)),
    ('decay_tau', (4e-3, 1e-3, 50e-3)),
    ('rise_power', (2.0, 'fixed')),
])
fit = psp.fit(y, x=x, xtol=1e-3, maxfev=100, params=params)


x = FitExplorer(fit=fit)
x.show()
