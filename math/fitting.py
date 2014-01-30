import numpy as _np
import grumpy as _gp
import pandas as _pd
import scipy.optimize as _spo


class FitFunction():

    def __init__(self, fitfunc):
        self.fitfunc = fitfunc

    def __call__(self, x_data, p):
        return self.fitfunc(_np.array(x_data), *p)

    def fit(self, p0, series, with_data=True):
        fitp = _gp.curve_fit(self.fitfunc, p0, series)[0]
        return FitObject(self.fitfunc, fitp, series, with_data)


class FitOptions():
    pass


class FitObject():

    def __init__(self, fitfunc, fitp, series, with_data=True):
        self.fitfunc = lambda x: fitfunc(fitp, x)
        self.fitp = fitp

        if with_data is True:
            self.data = _pd.DataFrame(series, columns=['raw'])
            self.data['fitted'] = self(series.index.values.
                                       astype(float))
            self.data['residuals'] = _np.abs(self.data['raw'] -
                                             self.data['fitted'])

    def __call__(self, x_data):
        return self.fitfunc(x_data)


def _plimit(func, pmin, pmax):
    pmin = _np.array(pmin)
    pmax = _np.array(pmax)

    def limited(x, *p):
        parray = _np.asarray(p)
        output = _gp.iterfy(func(x, *p))

        scale = []

        for index, params in enumerate(p):
            pcopy = parray.copy()
            pcopy[index] += .1
            dout = func(x, *tuple(pcopy))
            diff = _np.sum(_np.abs((output - dout) / (len(output) * output)))

            scale.append(1 / diff)

        lower = pmin - parray
        upper = parray - pmax

        lower[lower < 0] = 0
        upper[upper < 0] = 0

        output = output * (_np.prod(_np.exp(scale * lower)) *
                           _np.prod(_np.exp(scale * upper)))

        return output

    return limited


def _general_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def _weighted_general_function(params, xdata, ydata, function, weights):
    return weights * (function(xdata, *params) - ydata)


def curve_fit(f, pd_series, p0=None, pmin=None, pmax=None, weights=None, **kw):
    """
    Modified version of scipy's "curve_fit" wrapper for leastsq. Accepts a
    pandas Series instead of separate x and y arguments. Furthermore, optional
    "pmin" and "pmax" keyword parameters can be passed that attempt to bound
    the range over which leastsq can find solutions.
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
        f = f
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
        f = _plimit(f, pmin, pmax)

    args = (xdata, ydata, f)
    if weights is None:
        func = _general_function
    else:
        func = _weighted_general_function
        args += (_np.asarray(weights),)

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

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov
