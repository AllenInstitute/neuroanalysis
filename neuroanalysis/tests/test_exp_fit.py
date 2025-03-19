import numpy as np
import pytest

from neuroanalysis.data import TSeries
from neuroanalysis.fitting.exp import exp_decay, exp_fit, best_exp_fit_for_tau


@pytest.mark.parametrize('tau', 10**np.linspace(-4, 0, 10))
@pytest.mark.parametrize('yoffset', np.linspace(-0.1, 0.1, 3))
@pytest.mark.parametrize('yscale', 10**np.linspace(-4, -1, 4))
@pytest.mark.parametrize('yscale_sign', [-1, 1])
@pytest.mark.parametrize('fn', [exp_fit])  # , exp_fit
def test_ic_exp_fit(tau, yoffset, yscale, yscale_sign, fn, plot_errors=False, plot_all=False, raise_errors=True):
    noise = 5e-3
    duration = 0.2
    yscale *= yscale_sign

    _run_exp_fit_test(duration, fn, noise, plot_all, plot_errors, raise_errors, tau, yoffset, yscale)


@pytest.mark.parametrize('tau', 10**np.linspace(-4, 0, 10))
@pytest.mark.parametrize('yoffset', np.linspace(-1e-9, 1e-9, 3))
@pytest.mark.parametrize('yscale', 10**np.linspace(-13, -9, 4))
@pytest.mark.parametrize('yscale_sign', [-1, 1])
@pytest.mark.parametrize('fn', [exp_fit])  # , exp_fit
def test_vc_exp_fit(tau, yoffset, yscale, yscale_sign, fn, plot_errors=False, plot_all=False, raise_errors=True):
    noise = 50e-12
    duration = 0.02
    yscale *= yscale_sign

    _run_exp_fit_test(duration, fn, noise, plot_all, plot_errors, raise_errors, tau, yoffset, yscale)


def _run_exp_fit_test(duration, fn, noise, plot_all, plot_errors, raise_errors, tau, yoffset, yscale):
    rng = np.random.RandomState(0)
    sample_rate = 50e3
    params = {'yoffset': yoffset, 'yscale': yscale, 'tau': tau}
    fit, y = run_single_exp_fit(
        duration=duration,
        sample_rate=sample_rate,
        params=params,
        noise=noise,
        rng=rng,
        fit_func=fn,
    )
    if plot_all:
        plot_test_result(y, params, fit)
    try:
        check_exp_fit(y, params, fit, noise)
    except Exception:
        if plot_errors and not plot_all:
            plot_test_result(y, params, fit)
        if raise_errors:
            raise


def test_bad_curve(plot=False):
    params = {
        'yoffset': -278e-12 * (500/560) - 28e-12,
        'yscale': -234e-12 * (560/500),
        'tau': 4e-3,
    }
    noise = 1e-12
    duration = params['tau'] * 5  # * 6 and this will pass
    sample_rate = 1e5
    t = np.linspace(0, duration, int(duration * sample_rate))
    data = exp_decay(t, **params)
    data += np.random.normal(0, noise, data.shape)
    y = TSeries(data, time_values=t)
    fit = exp_fit(y)
    if plot:
        plot_test_result(y, params, fit)

    check_exp_fit(y, params, fit, noise)


def run_single_exp_fit(duration, sample_rate, params, noise, rng, fit_func):
    t = np.arange(0, duration, 1/sample_rate)
    offset = params['yoffset']
    scale = params['yscale']
    tau = params['tau']
    data = exp_decay(t, offset, scale, tau)
    data += rng.normal(0, noise, data.shape)
    y = TSeries(data, time_values=t)
    try:
        fit = fit_func(y)
    except Exception:
        print(f"Error fitting {fit_func} {params}")
        raise
    return fit, y


def check_exp_fit(y, params, fit, noise):
    # if fit['nrmse'] >= 0.05:
    #     raise AssertionError(f"Error too big: {fit['nrmse']}")
    fit_y = fit['model'](y.time_values)
    target_y = exp_decay(y.time_values, **params)
    fit['err_std'] = (target_y - fit_y).std()
    # assert np.allclose(fit['fit'], [params['yoffset'], params['yscale'], params['tau']], rtol=0.05)
    print(f"tau: {params['tau']} vs {fit['fit'][2]}")
    if fit['err_std'] >= noise * 0.3:
        raise AssertionError(f"Params: {params} Error too big: {fit['err_std']} >= {noise * 0.3}")


def calc_exp_error_curve(tau: float, data: TSeries):
    """Calculate the error surface for an exponential with *tau* and noisy *data* 
    """
    taus = tau * 10**np.linspace(-3, 3, 1000)
    errs = []
    for i in range(len(taus)):
        exp_y, err, yscale, yoffset = best_exp_fit_for_tau(taus[i], data.time_values, data.data)
        errs.append(err)
    return taus, errs


plot_window = None


def plot_test_result(y, params, fit):
    global plot_window
    import pyqtgraph as pg

    if plot_window is None:
        plot_window = pg.GraphicsLayoutWidget()
        plot_window.plt1 = plot_window.addPlot(0, 0)
        plot_window.plt2 = plot_window.addPlot(1, 0)
    plot_window.show()

    plt1 = plot_window.plt1
    plt1.addLegend()
    plt2 = plot_window.plt2

    plt1.plot(y.time_values, y.data, pen='w', label='data', name='data')
    plt1.setTitle(
        f"tau: {params['tau']:0.2g} yoffset: {params['yoffset']:0.2g} yscale: {params['yscale']:0.2g}"
        f" nrmse: {fit['nrmse']:0.2g}")  # err_std: {fit['err_std']:0.2g}")
    plt1.plot(y.time_values, fit['model'](y.time_values), pen='r', label='fit', name='fit')

    target_y = exp_decay(y.time_values, **params)
    plt1.plot(y.time_values, target_y, pen='b', label='target', name='target')

    taus, errs = calc_exp_error_curve(params['tau'], y)
    plt2.plot(taus, errs)
    plt2.addLine(x=params['tau'], pen='g')
    plt2.addLine(x=fit['fit'][2], pen='r')
    if 'memory' in fit:
        plt2.plot(list(fit['memory'].keys()), [m[2] for m in fit['memory'].values()], pen=None, symbol='o', symbolPen='r')
    pg.exec()


if __name__ == '__main__':
    test_bad_curve(plot=True)
    # test_ic_exp_fit(plot_all=False, plot_errors=True, raise_errors=False, fn=exact_fit_exp)
