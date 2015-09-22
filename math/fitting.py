import numpy as _np
import grumpy as _gp
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.optimize as _spo


class FitObject():

    _error = {0: 'Does not match number of input parameters',
              1: 'Does not match number of data points',
              2: 'Cannot calculate goodness of fit without errorbars'}

    def __init__(self, data, fitfunc, fitp, cov, p0, pmin, pmax, errorbars):
        self._fitfunc = fitfunc
        self._nargs, self._args = self._get_args()

        self._fitp = fitp
        self._cov = cov
        self._p0 = p0
        self._pmin = pmin
        self._pmax = pmax
        self._fitrange = None

        self._data = data
        self._residuals = self._calculate_residuals()
        self.errorbars = errorbars

    def __call__(self, x_data, params=None):
        if params is None:
            params = self.fitp
        return self.fitfunc(_np.asarray(x_data), *params)

    def __repr__(self):
        truncated = [x[:5] + ' = ' for x in self.args]
        params = ['{:3.3g}'.format(x) for x in self.fitp]
        errors = [' +/- ' + '{:1.1g}'.format(x) for x in self._perrors()]
        sigma = [' ({:1.0f})'.format(x) for x in self.fitp / self._perrors()]
        plist = ["".join(s) for s in zip(truncated, params, errors, sigma)]

        plist.append('\nchi2 = {:3.3g}'.format(self.chi2))
        if self.errorbars is not None:
            plist.append('Q = {:2.2g}'.format(self.Q))
        else:
            plist.append('S = %.3g' % (_np.sqrt(self.chi2) / len(self.data)))

        return str.join('\n', plist)

    def _get_args(self):
        import inspect
        args, varargs, varkw, defaults = inspect.getargspec(self.fitfunc)
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        if 'self' in args:
            return len(args)-2, args[2:]
        else:
            return len(args)-1, args[1:]

    def _calculate_residuals(self):
        xdata = self.fitdata.index.values.astype(float)
        res = self.fitdata - self.fitfunc(xdata, *self.fitp)
        return res

    def _chi2(self):
        if self._errorbars is None:
            e = 1
        else:
            e = self.errorbars

        return _np.sum((self.residuals / e) ** 2)

    def _Q(self):
        from scipy.stats import chi2 as _c2
        if self._errorbars is None:
            raise ValueError(self._error[3])

        return 1 - _c2.cdf(self._chi2(), len(self.fitdata) - self.nargs - 1)

    def _perrors(self):
        return _np.sqrt(_np.diag(self.cov))

    def _perror_projection(self, xs):
        import numpy.linalg as _linalg

        derrors, dvecs = _linalg.eig(self.cov)
        deviations = _np.sqrt(derrors) * dvecs

        bounds = []

        for i in range(len(derrors)):
            bounds.append(self.fitfunc(xs, *(self.fitp + deviations[:, i])))
            bounds.append(self.fitfunc(xs, *(self.fitp - deviations[:, i])))

        stacked = _np.vstack(bounds)
        lower = _np.min(stacked, axis=0)
        upper = _np.max(stacked, axis=0)

        return lower, upper

    def _maxerror_projection(self, xs):
        import numpy.linalg as _linalg

        derrors, dvecs = _linalg.eig(self.cov)
        dpmax = [_np.max(derrors[:] * dvecs[:, i]) for i in range(self.nargs)]

        lower = self.fitfunc(xs, *(self.fitp - _np.array(dpmax)))
        upper = self.fitfunc(xs, *(self.fitp + _np.array(dpmax)))

        return lower, upper

    def plot(self, xmin, xmax, rdens=30, type='linear', residuals=True):

        type_dict = {'linear': ['linear', 'linear'],
                     'semilogy': ['linear', 'log'],
                     'semilogx': ['log', 'linear'],
                     'loglog': ['log', 'log']}

        type_kwargs = {'linear': [{}, {}],
                       'semilogy': [{}, {'nonposy': 'mask'}],
                       'semilogx': [{'nonposx': 'mask'}, {}],
                       'loglog': [{'nonposx': 'mask'}, {'nonposy': 'mask'}]}

        if xmin > xmax:
            raise ValueError('xmin must be smaller than xmax')

        xscale = type_dict[type][0]
        xkwargs = type_kwargs[type][0]

        yscale = type_dict[type][1]
        ykwargs = type_kwargs[type][1]

        if not type in type_dict.keys():
            msg = '"type" not in "linear", "semilogy", "semilogx", or "loglog"'
            raise KeyError(msg)

        xs = self.data.index.values.astype(float)
        ys = self.data.values.astype(float)

        xf = self.fitdata.index.values.astype(float)
        yf = self.fitdata.values.astype(float)

        xr = _np.linspace(xmin, xmax, len(xs) * rdens)
        try:
            valid = _np.array([xs[xs > xmin][0], xs[xs < xmax][-1]])
        except:
            raise ValueError('No data in (xmin, xmax)!')
        yfit = self.fitfunc(xr, *self.fitp)
        space = _np.abs(_np.array([xmin, xmax]) - _np.array(valid))
        xlims = [xmin + .05 * space[0], xmax - .05 * space[0]]

        lower, upper = self._perror_projection(xr)

        f = _plt.figure()

        if residuals is True:
            ax = f.add_axes((0, 0, .7, .8))
            rax = f.add_axes((0, .875, .7, .125))
            yr = self.residuals.values.astype(float)
            if self.errorbars is not None:
                rax.errorbar(xf, yr, self.errorbars, fmt='.', alpha=.8)
            else:
                rax.plot(xf, yr, '.', alpha=.8)
            rax.axhline(0, alpha=.25)
            yloc = _plt.MaxNLocator(2)
            rax.yaxis.set_major_locator(yloc)
            rax.set_xscale(xscale, **xkwargs)
            rax.set_xlim(xlims)
            rax.xaxis.set_ticklabels([])
        else:
            ax = f.add_axes((0, 0, .7, 1))

        try:
            ax.fill_between(xr, lower, upper, alpha=.15)
        except:
            pass
        ax.plot(xr, yfit, alpha=.5)
        if self.errorbars is not None:
            ax.errorbar(xf, yf, self.errorbars, fmt='.', alpha=.8)
        ax.plot(xs, ys, '.', alpha=.8)

        if self.fitrange is not None:
            ax.axvspan(*self.fitrange, alpha=.03)

        f.text(.725, .5, self._plotstr())

        ax.set_xscale(xscale, **xkwargs)
        ax.set_yscale(yscale, **ykwargs)
        ax.set_xlim(xlims)

        return f, ax

    def _plotstr(self):
        truncated = [x[:5] + ' = ' for x in self.args]
        params = ['{:3.3g}'.format(x) for x in self.fitp]
        errors = ['$\pm$' + '{:1.1g}'.format(x) for x in self._perrors()]
        sigma = [' ({:1.0f})'.format(x) for x in self.fitp / self._perrors()]
        plist = ["".join(s) for s in zip(truncated, params, errors, sigma)]

        plist.append('\n$\chi^2$ = {:3.3g}'.format(self.chi2))
        if self.errorbars is not None:
            plist.append('Q = {:2.2g}'.format(self.Q))
        else:
            plist.append('$\sigma_{res}$ = %.3g' % (_np.sqrt(self.chi2)
                         / len(self.data)))

        return str.join('\n', plist)

    def coarsen(self, points, method=_np.mean):
        pass

    def apply(self):
        pass

    def reset_data(self):
        self.fitdata = self.data.copy()

    def fit(self):
        newp, newcov = curve_fit(self.fitdata, self.fitfunc, self.p0,
                                 pmin=self.pmin, pmax=self.pmax, fitobj=False,
                                 errorbars=self.errorbars)
        self._fitp = newp
        self._cov = newcov
        self._residuals = self._calculate_residuals()
        return self

    @property
    def data(self):
        return self._data

    @property
    def fitdata(self):
        if self.fitrange is None:
            return self._data
        else:
            return self._data.ix[self.fitrange[0]:self.fitrange[1]]

    @property
    def residuals(self):
        return self._residuals

    @property
    def errorbars(self):
        if (self._errorbars is None) or (self.fitrange is None):
            return self._errorbars
        else:
            return self._errorbars.ix[self.fitrange[0]:self.fitrange[1]]

    @errorbars.setter
    def errorbars(self, newerrors):
        if newerrors is None:
            self._errorbars = None
        elif len(newerrors) == 1:
            self._errorbars = _pd.Series([newerrors] * len(self.data),
                                         index=self.data.index)
        elif len(newerrors) != len(self.data):
            raise ValueError(self._error[1])
        else:
            self._errorbars = _pd.Series(newerrors, index=self.data.index)
    pass

    @errorbars.deleter
    def errorbars(self):
        self._errorbars = None

    def remove_errorbars(self):
        del(self.errorbars)
        pass

    @property
    def chi2(self):
        return self._chi2()

    @property
    def Q(self):
        return self._Q()

    @property
    def fitfunc(self):
        return self._fitfunc

    @property
    def nargs(self):
        return self._nargs

    @property
    def args(self):
        return self._args

    @property
    def fitp(self):
        return self._fitp

    @property
    def cov(self):
        return self._cov

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, newp):
        if len(newp) != self.nargs:
            raise ValueError(self._error[0])
        self._p0 = newp

    @property
    def pmin(self):
        return self._pmin

    @pmin.setter
    def pmin(self, newpmin):
        if len(newpmin) != self.nargs:
            raise ValueError(self._error[0])
        self._pmin = newpmin

    @property
    def pmax(self):
        return self._pmax

    @pmax.setter
    def pmax(self, newpmax):
        if len(newpmax) != self.nargs:
            raise ValueError(self._error[0])
        self._pmax = newpmax

    @property
    def fitrange(self):
        return self._fitrange

    @fitrange.setter
    def fitrange(self, limits):
        if _np.squeeze(_np.asarray(limits)).shape[0] == 2:
            self._fitrange = tuple(limits)

    @fitrange.deleter
    def fitrange(self):
        self._fitrange = None

    def zoom(self, axis=None):
        if axis is None:
            self.fitrange = _plt.gca().get_xlim()
        else:
            self.fitrange = axis.get_xlim()

    def set_fitrange(self, xmin, xmax):
        self.fitrange = (xmin, xmax)

    def reset_fitrange(self):
        del(self.fitrange)
        pass


def _plimit(func, pmin, pmax):
    pmin = _np.asarray(pmin)
    pmax = _np.asarray(pmax)

    def limited(x, *p):
        parray = _np.asarray(p)

        if any(parray > pmax):
            return _np.inf
        elif any(parray < pmin):
            return _np.inf

        return func(x, *p)

    return limited


def _general_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def _weighted_general_function(params, xdata, ydata, function, errorbars):
    return (function(xdata, *params) - ydata) / errorbars


def curve_fit(pd_series, f, p0=None, pmin=None, pmax=None,
              errorbars=None, fitobj=True, **kw):
    """
    Modified version of scipy's "curve_fit" wrapper for leastsq. Accepts a
    pandas Series instead of separate x and y arguments. Additionally,
    optional "pmin" and "pmax" keyword parameters can be passed that attempt
    to bound the range over which leastsq can find solutions.
    """

    xdata = pd_series.index.values.astype(float)
    ydata = pd_series.values.astype(float)

    if p0 is None:
        # determine number of parameters by inspecting the function
        import inspect
        args, varargs, varkw, defaults = inspect.getargspec(f)
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        if 'self' in args:
            p0 = [1.0] * (len(args)-2)
        else:
            p0 = [1.0] * (len(args)-1)

    p0 = _np.asarray(_gp.iterfy(p0))

    if (pmin is None) & (pmax is None):
        ff = f
    else:
        if pmin is None:
            pmin = _np.asarray([-_np.inf] * len(p0))
            pmax = _np.asarray(pmax)
            p0[p0 > pmax] = pmax[p0 > pmax]
        elif pmax is None:
            pmax = _np.asarray([_np.inf] * len(p0))
            pmin = _np.asarray(pmin)
            p0[p0 < pmin] = pmin[p0 < pmin]
        else:
            pmin = _np.asarray(pmin)
            pmax = _np.asarray(pmax)
            p0[p0 > pmax] = (pmax[p0 > pmax] + pmin[p0 > pmax]) / 2
            p0[p0 < pmin] = (pmax[p0 < pmin] + pmin[p0 < pmin]) / 2
        ff = _plimit(f, pmin, pmax)

    args = (xdata, ydata, ff)
    if errorbars is None:
        func = _general_function
    else:
        func = _weighted_general_function
        args += (_np.asarray(errorbars),)

    # Remove full_output from kw, otherwise we're passing it in twice.
    return_full = kw.pop('full_output', False)
    res = _spo.leastsq(func, p0, args=args, full_output=1, **kw)
    (popt, pcov, infodict, errmsg, ier) = res

    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)

    if (len(ydata) > len(p0)) and pcov is not None:
        s_sq = (func(popt, *args)**2).sum()/(len(ydata)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = _np.inf

    if fitobj:
        return FitObject(pd_series, f, popt, pcov, p0, pmin, pmax, errorbars)

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov
