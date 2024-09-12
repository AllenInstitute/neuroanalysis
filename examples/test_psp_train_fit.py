import numpy as np

import pyqtgraph as pg
from neuroanalysis.fitting import PspTrain
from neuroanalysis.ui.fitting import FitExplorer

pg.dbg()

pg.mkQApp()

t = np.arange(2000) * 1e-4

rise_time = 5e-3
decay_tau = 20e-3
n_psp = 4
args = {
    'yoffset': (0, 'fixed'),
    'xoffset': (2e-3, -1e-3, 5e-3),
    'rise_time': (rise_time, rise_time * 0.5, rise_time * 2),
    'decay_tau': (decay_tau, decay_tau * 0.5, decay_tau * 2),
    'rise_power': (2, 'fixed'),
}
for i in range(n_psp):
    args['xoffset%d' % i] = (25e-3 * i, 'fixed')
    args['amp%d' % i] = (250e-6, 0, 10e-3)

fit_kws = {'xtol': 1e-4, 'maxfev': 1000, 'nan_policy': 'omit'}
model = PspTrain(n_psp)

args2 = {k: (v[0] if isinstance(v, tuple) else v) for k, v in args.items()}
y = np.random.normal(size=len(t), scale=30e-6) + model.eval(x=t, **args2)

fit = model.fit(y, x=t, params=args, fit_kws=fit_kws, method='leastsq')

ex = FitExplorer(fit)
ex.show()
