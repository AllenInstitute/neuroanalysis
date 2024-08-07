import numpy as np

from neuroanalysis.data import TSeries
from neuroanalysis.fitting.exp import exp_decay, exp_fit, fit_with_explicit_hessian, exact_fit_exp


def test_bad_curve(plot=False):
    tau = 4e-3
    scale = -234e-12 * (560/500)
    offset = -278e-12 * (500/560) - 28e-12
    t = np.linspace(0, tau * 4, 1000)
    data = exp_decay(t, offset, scale, tau)
    data += np.random.normal(0, 1e-12, data.shape)
    y = TSeries(data, time_values=t)

    # fit = exp_fit(y)
    # fit = fit_with_explicit_hessian(y)
    fit = exact_fit_exp(y)

    plot_if_needed(fit, plot, y)
    assert fit['nrmse'] < 0.05, f"Error too big: {fit['nrmse']}"
    assert np.allclose(fit['fit'], [offset, scale, tau], rtol=0.05)


def plot_if_needed(fit: dict, plot: bool, y: TSeries):
    if plot:
        import pyqtgraph as pg
        plt = pg.plot(y.time_values, y.data, pen='w')
        plt.setTitle(fit['nrmse'])
        plt.plot(y.time_values, fit['model'](y.time_values), pen='r')
        pg.exec()


if __name__ == '__main__':
    test_bad_curve(plot=True)
