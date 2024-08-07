import numpy as np

from neuroanalysis.data import TSeries
from neuroanalysis.fitting.exp import exp_decay, exp_fit, fit_with_explicit_hessian


def test_exp_fit(plot=False, raise_errors=True):
    rng = np.random.RandomState(0)
    duration = 0.1
    sample_rate = 50e3
    taus = 10**np.linspace(-4, 0, 10)
    for mode in ('ic', 'vc'):
        if mode == 'ic':
            yoffsets = np.linspace(-0.1, 0.1, 5)
            yscales = 10**np.linspace(-4, -1, 10)
            yscales = np.concatenate([yscales, -yscales])
            noise = 5e-3
        else:
            yoffsets = np.linspace(-1e-9, 1e-9, 5)
            yscales = 10**np.linspace(-13, -9, 10)
            yscales = np.concatenate([yscales, -yscales])
            noise = 50e-12
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
                    fit_func=exp_fit,
                )
                try:
                    check_exp_fit(y, params, fit, noise)
                except Exception:
                    if plot:
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
    duration = params['tau'] * 4
    t = np.linspace(0, duration, 1000)
    data = exp_decay(t, **params)
    data += np.random.normal(0, 1e-12, data.shape)
    y = TSeries(data, time_values=t)
    fit = exp_fit(y)
    fit, y = run_single_exp_fit(
        duration=duration,
        sample_rate=1000 / duration,
        params=params,
        noise=noise,
        rng=np.random.RandomState(1234),
        fit_func=exp_fit,
    )

    # fit = fit_with_explicit_hessian(y)
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
    assert (target_y - fit_y).std() < noise * 0.3
    # assert np.allclose(fit['fit'], [params['yoffset'], params['yscale'], params['tau']], rtol=0.05)


def plot_test_result(y, params, fit):
    import pyqtgraph as pg
    plt = pg.plot(y.time_values, y.data, pen='w')
    plt.setTitle(f"tau: {params['tau']:0.2g} yoffset: {params['yoffset']:0.2g} yscale: {params['yscale']:0.2g} nrmse: {fit['nrmse']}")
    plt.plot(y.time_values, fit['model'](y.time_values), pen='r')
    pg.exec()


if __name__ == '__main__':
    # test_bad_curve(plot=True)
    test_exp_fit(plot=True, raise_errors=False)
