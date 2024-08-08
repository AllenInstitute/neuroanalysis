import numpy as np

from neuroanalysis.data import TSeries
from neuroanalysis.fitting.exp import exp_decay, exp_fit, exact_fit_exp, test_tau


def test_exp_fit(plot_errors=False, plot_all=False, raise_errors=True, fn=exp_fit):
    rng = np.random.RandomState(0)
    duration = 0.1
    sample_rate = 50e3
    taus = 10**np.linspace(-4, 0, 10)
    for mode in ('ic', 'vc'):
        if mode == 'ic':
            yoffsets = np.linspace(-0.1, 0.1, 5)
            # yscales = 10**np.linspace(-4, -1, 10)
            yscales = 10**np.linspace(-1, 0, 10)
            noise = 5e-3
        else:
            yoffsets = np.linspace(-1e-9, 1e-9, 5)
            yscales = 10**np.linspace(-13, -9, 10)
            noise = 50e-12
        yscales = np.concatenate([yscales, -yscales])
        for tau in taus:
            for yoffset in yoffsets:
                for yscale in yscales:
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


def test_exact_fit_exp():
    test_exp_fit(fn=exact_fit_exp)


def test_bad_curve(plot=False):
    params = {
        'yoffset': -278e-12 * (500/560) - 28e-12,
        'yscale': -234e-12 * (560/500),
        'tau': 4e-3,
    }
    noise = 1e-12
    duration = params['tau'] * 4
    t = np.linspace(0, duration, 1000)
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
    # fit = fit_with_explicit_hessian(y)
    return fit, y


def check_exp_fit(y, params, fit, noise):
    # assert fit['nrmse'] < 0.05, f"Error too big: {fit['nrmse']}"
    fit_y = fit['model'](y.time_values)
    target_y = exp_decay(y.time_values, **params)
    fit['err_std'] = (target_y - fit_y).std()
    assert fit['err_std'] < noise * 0.3
    # assert np.allclose(fit['fit'], [params['yoffset'], params['yscale'], params['tau']], rtol=0.05)


def calc_exp_error_curve(tau:float, data:TSeries):
    """Calculate the error surface for an exponential with *tau* and noisy *data* 
    """
    taus = tau * 10**np.linspace(-3, 3, 1000)
    errs = []
    for i in range(len(taus)):
        exp_y, err, yscale, yoffset = test_tau(taus[i], data.time_values, data.data)
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

    plt1 = plot_window.plt1
    plt2 = plot_window.plt2

    plt1 = pg.plot(y.time_values, y.data, pen='w', label='data')
    plt1.setTitle(f"tau: {params['tau']:0.2g} yoffset: {params['yoffset']:0.2g} yscale: {params['yscale']:0.2g}"
                 f" nrmse: {fit['nrmse']:0.2g} err_std: {fit['err_std']:0.2g}")
    plt1.plot(y.time_values, fit['model'](y.time_values), pen='r', label='fit')

    target_y = exp_decay(y.time_values, **params)
    plt1.plot(y.time_values, target_y, pen='b', label='target')
    plt1.addLegend()

    taus, errs = calc_exp_error_curve(params['tau'], y)
    plt2.plot(taus, errs)
    plt2.addLine(x=params['tau'], pen='g')
    plt2.addLine(x=fit['fit'][2], pen='r')
    plt2.plot(list(fit['memory'].keys()), [m[2] for m in fit['memory'].values()], pen=None, symbol='o', symbolPen='r') 
    pg.exec()


if __name__ == '__main__':
    # test_bad_curve(plot=True)
    test_exp_fit(plot_all=False, plot_errors=True, raise_errors=False, fn=exact_fit_exp)
