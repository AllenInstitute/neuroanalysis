from collections import OrderedDict
import numpy as np
import scipy.stats
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import lmfit.minimizer 


class FitExplorer(QtWidgets.QWidget):
    def __init__(self, fit=None, model=None, data=None, args=None):
        QtWidgets.QWidget.__init__(self)
        self.model = model if model is not None else fit.model
        self.args = args
        self.data = data
        self.fit = None
        
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        self.ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.splitter.addWidget(self.ptree)
        self.plot = pg.PlotWidget()
        self.splitter.addWidget(self.plot)
        
        self.params = pg.parametertree.Parameter.create(name='param_root', type='group',
            children=[
                dict(name='initial parameters', type='group'),
                dict(name='constraints', type='group'),
                dict(name='fit', type='action'),
                dict(name='fit method', type='list', values=['leastsq', 'least_squares', 'brute'] + list(lmfit.minimizer.SCALAR_METHODS.keys())),
                dict(name='fit parameters', type='group'),
                dict(name='chi squared', type='float', readonly=True),
                dict(name='normalized RMS error', type='float', readonly=True),
                dict(name='pearson r', type='float', readonly=True),
                dict(name='pearson p', type='float', readonly=True),
            ])
        
        for k in self.model.param_names:
            for pname in ['initial parameters', 'fit parameters']:
                p = pg.parametertree.Parameter.create(name=k, type='float', value=0, dec=True, minStep=1e-12, step=0.2, delay=0)
                self.params.child(pname).addChild(p)
                
            p = ConstraintParameter(self.params.child('initial parameters', k))
            self.params.child('constraints').addChild(p)
                
            
        if fit is not None:
            self.set_fit(fit)
            
        self.ptree.setParameters(self.params, showTop=False)

        for k in self.model.param_names:
            for pname in ['initial parameters', 'fit parameters']:
                param = self.params.child(pname, k)
                spin = list(param.items.keys())[0].widget
                spin.proxy.setDelay(0)
        
        self.update_plots()
        
        self.params.sigTreeStateChanged.connect(self.update_plots)
        self.params.child('fit').sigActivated.connect(self.refit)

    def set_fit(self, fit, update_params=True):
        self.fit = fit
        if update_params:
            self.data = fit.data
            self.args = fit.userkws
            self.model = fit.model
            self._fill_init_params(self.params.child('initial parameters'), self.fit)
            self._fill_constraints(self.params.child('constraints'), self.fit)
        self._fill_params(self.params.child('fit parameters'), self.fit)
        self._update_fit_stats()
        
    def _update_fit_stats(self):
        with pg.SignalBlock(self.params.sigTreeStateChanged, self.update_plots):
            self.params['chi squared'] = self.fit.chisqr

            args = self.args.copy()
            args.update(self.fit_params())
            fity = self.model.eval(y=self.data, **args)
            residual = self.data - fity
            rmse = (residual**2 / residual.size).sum() ** 0.5
            self.params['normalized RMS error'] = rmse / self.data.std()
            
            r, p = scipy.stats.pearsonr(self.data, fity)
            self.params['pearson r'] = r
            self.params['pearson p'] = p
        
    def update_plots(self):
        self.plot.clear()

        args = self.args.copy()
        self.plot.plot(args['x'], self.data, antialias=True)

        args.update(self.initial_params())
        self.plot.plot(args['x'], self.model.eval(y=self.data, **args), pen='y', antialias=True)

        if self.fit is not None:
            args.update(self.fit_params())
            fity = self.model.eval(y=self.data, **args)
            self.plot.plot(args['x'], fity, pen='g', antialias=True)
            if hasattr(self.fit, 'eval_uncertainty'):
                # added in 0.9.6
                try:
                    err = self.fit.eval_uncertainty()
                    c1 = pg.PlotCurveItem(args['x'], fity-err)
                    c2 = pg.PlotCurveItem(args['x'], fity+err)
                    fill = pg.FillBetweenItem(c1, c2, pg.mkBrush((0, 255, 0, 50)))
                    self.plot.addItem(fill, ignoreBounds=True)
                    fill.setZValue(-1)
                except Exception:
                    # eval_uncertainty is broken in some versions
                    pass
            self._update_fit_stats()

    def fit_params(self):
        params = OrderedDict()
        for ch in self.params.child('fit parameters').children():
            params[ch.name()] = ch.value()
        return params
        
    def initial_params(self):
        params = OrderedDict()
        for ch in self.params.child('initial parameters').children():
            params[ch.name()] = ch.value()
        return params

    def constraints(self):
        params = OrderedDict()
        for ch in self.params.child('constraints').children():
            params[ch.name()] = ch.constraint()
        return params

    def refit(self):
        args = self.args.copy()
        params = self.constraints()
        self.set_fit(self.model.fit(self.data, method=self.params['fit method'], params=params, **args), update_params=False)
        
    def _fill_params(self, root, fit):
        for k in self.model.param_names:
            root[k] = fit.params[k].value

    def _fill_init_params(self, root, fit):
        for k in self.model.param_names:
            root[k] = fit.params[k].init_value

    def _fill_constraints(self, root, fit):
        for k in self.model.param_names:
            root.child(k).set_from_param(fit.params[k])


class ConstraintParameter(pg.parametertree.parameterTypes.GroupParameter):
    def __init__(self, source_param):
        self.source = source_param
        pg.parametertree.parameterTypes.GroupParameter.__init__(self, name=self.source.name(), children=[
            {'name': 'type', 'type': 'list', 'values': ['unconstrained', 'range', 'fixed']},
            {'name': 'min', 'type': 'float', 'visible': False, 'dec': True, 'minStep': 1e-12},
            {'name': 'max', 'type': 'float', 'visible': False, 'dec': True, 'minStep': 1e-12},
        ])
        
        self.child('type').sigValueChanged.connect(self.type_changed)
        
    def type_changed(self):
        isrange = self['type'] == 'range'
        self.child('min').setOpts(visible=isrange)
        self.child('max').setOpts(visible=isrange)

    def constraint(self):
        sval = self.source.value()
        if self['type'] == 'unconstrained':
            return sval
        elif self['type'] == 'range':
            return (sval, self['min'], self['max'])
        else:
            return (sval, 'fixed')
        
    def set_constraint(self, c):
        if isinstance(c, tuple):
            if c[1] == 'fixed':
                self['type'] = 'fixed'
            else:
                self['type'] = 'range'
                self['min'] = c[1]
                self['max'] = c[2]
        else:
            self['type'] = 'unconstrained'

    def set_from_param(self, param):
        if not param.vary:
            self['type'] = 'fixed'
        else:
            if not np.isfinite(param.min) and not np.isfinite(param.max):
                self['type'] = 'unconstrained'
            else:
                self['type'] = 'range'
                self['min'] = param.min
                self['max'] = param.max
