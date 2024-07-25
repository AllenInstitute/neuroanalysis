import numpy as np

from neuroanalysis.data import TSeries
from neuroanalysis.fitting.exp import exp_decay, exp_fit


def test_bad_curve(plot=False):
    tau = 4e-3
    scale = -234e-12 * (560/500)
    offset = -278e-12 * (500/560) - 28e-12
    t = np.linspace(0, tau * 4, 1000)
    data = exp_decay(t, offset, scale, tau)
    data += np.random.normal(0, 1e-12, data.shape)
    y = TSeries(data, time_values=t)
    fit = exp_fit(y)
    if plot:
        import pyqtgraph as pg
        plt = pg.plot(y.time_values, y.data, pen='w')
        plt.setTitle(fit['nrmse'])
        plt.plot(y.time_values, fit['model'](y.time_values), pen='r')
        pg.exec()
    assert fit['nrmse'] < 0.05, f"Error too big: {fit['nrmse']}"
    assert np.allclose(fit['fit'], [offset, scale, tau], rtol=0.05)


if __name__ == '__main__':
    test_bad_curve(plot=True)
    print("All tests passed")