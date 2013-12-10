import numpy as _np
import grumpy as _gp
import pandas as _pd
import cylowess as _cl
import scipy as _sp
import scipy.stats as _sps
import scipy.fftpack as _fft
import scipy.signal as _spsig
import scipy.optimize as _spo
import scipy.integrate as _spi


def clean_series(pd_series):

    """
    Fills NaN values in a pandas Series with linearly interpolated values.

    If the first values are NaN, they are replaced with the first valid value
    in the Series. clean_series also outputs a mask with the locations of all
    NaNs. This is used to coerce the output array to be consistent with the
    input array. clean_series is used to prepare pandas Series for various
    math functions that may break with NaN values.

    Accepts a pandas Series, returns a pandas Series and boolean array with
    the same shape.

    WARNING: Interpolation and backfilling of NaNs is limited to 5
    consecutive data points. More missing data will result is NaNs being left
    in an array, potentially causing errors.

    Parameters
    ----------
    kwargs : keywords
        Passed to pandas Series.interpolate method
    """

    nanmask = _np.isnan(pd_series.values.astype(float))
    clean_data = pd_series.interpolate(method='slinear')
    clean_data = clean_data.fillna(method='bfill')

    return clean_data, nanmask


def force_spacing(pd_series):

    """
    Interpolates input series between the smallest non-NaN index and largest
    non-NaN index at uniform spacing.

    This function is used to condition data that needs to be uniformly spaced
    for various transforms (Fourier, etc). Endpoint NaNs are dropped and
    linear interpolation is done between them. Endpoint NaNs cannot be
    properly interpolated and back-filled or forward-filled data can cause
    transform artifacts.

    Accepts a pandas Series, returns a pandas Series of the same length.
    """

    if index_uniformity_of(pd_series) is True:
        return clean_series(pd_series)[0]

    data_length = len(pd_series)
    data_indices = pd_series.dropna().index.values.astype(float)
    data_points = pd_series.dropna().values.astype(float)

    small_index = _np.min(data_indices)
    big_index = _np.max(data_indices)
    forced_indices = _np.linspace(small_index, big_index, data_length)

    forced_data = _sp.interp(forced_indices, data_indices, data_points)

    return _pd.Series(forced_data, index=forced_indices)


def index_uniformity_of(pd_series):

    """
    Checks if the indices of a pandas Series is uniformly spaced.

    Accepts a pandas Series, returns boolean True or False.
    """

    data_indices = pd_series.index.values.astype(float)
    spacing = _np.diff(data_indices)
    if all(spacing - _np.mean(spacing) < .01 * _np.mean(spacing)):
        return True
    else:
        return False


def reject_outliers(data, deviation_tolerance=.6745):

    """
    Replaces outliers in a numpy array with NaN.

    Checks for outliers by comparing the distance of each element in the
    input array from the median value of the input array. The median of
    these distances is the interquartile range (IQR).

    If the distance of a point in the input array from the median is larger
    than some specified multiple of the IQR, the point is discarded and
    replaced is NaN. The multiple can be controlled using the
    deviation_tolerance parameter, the threshold for discarding outliers is
    deviation_tolerance / .6745. The numeric factor normalizes the IQR such
    that a deviation_tolerance of 1 is equivalent to 1 standard deviation
    for gaussian input data.

    Accepts numpy array, returns numpy array of the same length.

    Parameters
    ----------
    deviation_tolerance : value, default .6745
        deviation_tolerance sets the threshold for whether an element from
        the input vector is removed by the outlier rejection. The default
        value is equivalent to selecting only the interquartile range.
    """

    data = data.astype('float')
    distance = _np.abs(data - _np.median(data))
    sigma = _np.median(distance) / .6745
    data[distance > deviation_tolerance * sigma] = _np.nan
    return data


def robust_mean(data, stdcutoff=None, **kwargs):

    """
    Robustified mean. Rejects outliers before taking mean. Ignores NaNs.

    Significant speedup when utilizing the "stdcutoff" argument.

    Accepts numpy array, returns value.

    Parameters
    ----------
    stdcutoff : value, default None
        stdcutoff is compared to the std. dev. of input data. Explicit
        outlier rejection only occurs if this test fails. 10x+ speedup.

    **kwargs passed to reject_outliers
    deviation_tolerance : value, default .6745
        Threshold for outlier rejection normalized such that 1 is
        equivalent to the 1 standard deviation for gaussian input data.
        The default value of .6745 will output the interquartile mean.
    """

    if len(data) <= 2:
        return _sps.nanmean(data)

    if stdcutoff is None:
        return _sps.nanmean(reject_outliers(data, **kwargs))
    else:
        if _np.std(data) < stdcutoff:
            return _sps.nanmean(data)
        else:
            return _sps.nanmean(reject_outliers(data, **kwargs))


def polysmooth(series, order=5):

    """
    Fits a polynomial to a pandas Series.

    Outputs the fitted polynomial evaluated at the index values where the
    input series value is finite. NaN everywhere else.

    Accepts (single index) pandas Series, returns a numpy array of the same
    length.

    Parameters
    ----------
    order : integer, default 5
        Order of polynomial fitting.
    """

    output = _np.empty(len(series))
    output[:] = _np.nan
    nanmask = _np.isnan(series.values.astype(float))
    x_data = series.dropna().index.values.astype(float)
    y_data = series.dropna().values.astype(float)
    fit_polynomial = _np.polyfit(x_data, y_data, order)
    output[~nanmask] = _np.polyval(fit_polynomial, x_data)
    return output


def lowess(series, frac=.5, delta=None, it=None):

    """
    Smooths a pandas Series using local linear regression (lowess).

    grumpy.lowess implements the fast 'cylowess' implementation built by
    C. Vogel. The original code can be found here:
    https://github.com/carljv/Will_it_Python/

    Outputs the smoothed function at the index values where the
    input series value is finite. NaN everywhere else.

    Accepts (single index) pandas Series, returns a numpy array of the same
    length.

    Parameters
    ----------
    frac : float, default .5
            Between 0 and 1. The fraction of the data used
            when estimating each y-value.
    it : int, default 3
            The number of residual-based reweightings
            to perform. These are expensive but robusitfy the regression.
    delta : float, default 1 percent of range(x_data)
            Distance within which to use linear-interpolation
            instead of weighted regression.
    """

    output = _np.empty(series.shape)
    output[:] = _np.nan
    nanmask = _np.isnan(series.values.astype(float))
    x_data = series.dropna().index.values.astype(float)
    y_data = series.dropna().values.astype(float)

    if it is None:
        it = 3
    if delta is None:
        delta = .01 * (max(x_data) - min(x_data))

    smooth = _cl.lowess(y_data, x_data, frac=frac, it=it, delta=delta)
    output[~nanmask] = smooth[:, 1]
    return output


def rolling_window(series, window_length=5, window='hanning',
                   delay_compensation=True):

    """
    Smooths a pandas series using a moving windowed average.

    Generates a window of type 'window' and length 'window_length' and
    convolves this window with the input pandas series. By default, this
    convolution is run in both directions and averaged to compensate for
    the time lag introduced by rolling window techniques and also doubles the
    smoothing power (generally set by the window length). The
    delay-compensated rolling window filter is non-causal.

    Accepts a (single index) pandas series, returns a numpy array of the
    same length.

    Parameters
    ----------
    window_length : int, default 5
        Number of datapoints included in smoothing window.
    window : ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
        Window type. Flat is a standard moving average.
    delay_compensation : boolean, default True
        Flag that turns on delay compensation. Doubles smoothing power and
        removes time lag associated with moving averages at cost of being
        a non-causal filter.
    """

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is not 'flat', 'hanning', 'hamming', \
              'bartlett', or 'blackman'")

    nanmask = _np.isnan(series.values.astype(float))
    x_data = series.fillna(method='bfill').values.astype(float)

    if window == 'flat':
        w = _np.ones(window_length, 'd')
    else:
        w = eval('_np.' + window + '(window_length)')

    s = _np.r_[x_data[window_length - 1:0:-1], x_data,
               x_data[-1:-window_length:-1]]

    smoothed = _np.convolve(w / w.sum(), s, mode='valid')
    smoothed = smoothed[(window_length - 1):]
    smoothed[nanmask] = _np.nan

    if delay_compensation is True:

        x_reversed = x_data[::-1]
        s_r = _np.r_[x_reversed[window_length - 1:0:-1], x_reversed,
                     x_reversed[-1:-window_length:-1]]

        smoothed_r = _np.convolve(w / w.sum(), s_r, mode='valid')
        smoothed_r = smoothed_r[(window_length - 1):][::-1]
        smoothed_r[nanmask] = _np.nan

        return (smoothed + smoothed_r) / 2

    else:

        return smoothed


def correlate(series_a, series_b, as_series=False):

    """
    Performs a rolling cross-correlation on two pandas Series.

    This is a pandas-specific wrapper for scipy.correlate. The pandas
    Series are passed to "clean_series" before being passed to sp.correlate
    to fill missing values. The output of correlate is the same length as the
    shortest input array. "correlate" also has an optional "as_series" flag
    to output a series with the rolling offset as the indices.

    Accepts two pandas Series, outputs a numpy array or pandas Series.

    Parameters
    ----------
    as_series : boolean, default False
        Outputs a series with points of offset as the indices
    """

    clean_a = clean_series(series_a)[0]
    clean_b = clean_series(series_b)[0]

    index_values = clean_a.index.values.astype(float)
    index_range = _np.max(index_values) - _np.min(index_values)
    norm_factor = index_range / len(index_values)

    correlated = norm_factor * _sp.correlate(clean_a, clean_b, 'same')

    if as_series is False:
        return correlated.values.astype(float)
    else:
        indices = _np.arange(_np.max((len(series_a), len(series_b))))
        center = _np.ceil(_np.median(indices))
        offset = indices - center
        return _pd.Series(correlated.values.astype(float), index=offset)


def corr_df(pd_dframe, reference=None, series_list=None):

    """
    Cross-correlates columns of a pandas DataFrame.

    Calls the grumpy.pdmath.correlate function to cross-correlate columns
    of the dataframe. With no keyword arguments passed to the function,
    corr_df correlates adjacent columns based on their position in the
    DataFrame.

    If a reference column is passed, all other columns will be
    cross-correlated against the reference column.

    If a list of series is passed, only a subset of the DataFrame is used.

    Parameters
    ----------
    reference : value, default None
        A name to specificy a reference column.
    series_list : list, default None
        A list of names to specify a subset of the DataFrame columns.
    """

    df = pd_dframe
    new_cols = []

    if series_list is None:
        names = pd_dframe.columns.values
    else:
        names = _np.array(series_list)

    if reference is not None:
        mask = _np.array([reference == name for name in names])
        series_list = names[~mask]

        for loop_col in series_list:
            new_cols.append(correlate(df[reference], df[loop_col],
                            as_series=True))

        return _pd.DataFrame(dict(zip(names, new_cols)))

    else:

        for index, loop_col in enumerate(names[1:]):
            new_cols.append(correlate(df[names[index]], df[loop_col],
                            as_series=True))

        return _pd.DataFrame(dict(zip(names[1:], new_cols)))


def fft(series):

    """
    Calculates the (single-sided) FFT of a pandas Series.

    Internally absfft calls scipy's 'fft' function and 'fftfreq' helper
    functions. absfft returns The fft of the input series indexed
    by the transformed frequency scale generated by 'fftfreq'. WARNING:
    absfft assumes that the spacing of the input series is uniform and
    passes the mean of the index spacing to 'fftfreq'. In the future I may
    improve this function by interpolating the input data.

    absfft returns only the positive frequency values. NaNs are returned for
    the negative frequency values. However, the negative frequency indexes
    are preserved and the length of the output matches the length of the
    input for consistency.

    Accepts (single index) pandas Series, returns a pandas Series of the
    same length.
    """

    output = _np.zeros(len(series)) + 0j

    data = force_spacing(series)
    x_data = data.index.values.astype(float)
    y_data = data.values.astype(float)

    fft_length = _np.ceil(len(data) / 2)
    fft_spacing = _np.mean(_np.diff(x_data))
    fft_range = max(x_data) - min(x_data)

    fft_freq = _fft.fftfreq(len(series), d=fft_spacing)
    fft_mag = _fft.fft(y_data)
    fft_mag[-fft_length:] = _np.nan

    output[:len(fft_mag)] = fft_mag
    output[len(fft_mag):] = _np.nan

    return _pd.Series(_np.sqrt(2 * fft_range) / len(x_data)
                      * output, index=fft_freq)


def absfft(series):

    """
    Calculates the (single-sided) FFT magnitude of a pandas Series.
    """

    return _np.abs(fft(series))


def psd(series):

    """
    Calculates the (single-sided) PSD of a pandas Series.
    """

    return absfft(series) ** 2


def hilbert(pd_series):

    data = force_spacing(pd_series)
    return _spsig.hilbert(data)


def hilbert_amplitude(pd_series):

    return (hilbert(pd_series) + pd_series).apply(_np.abs)


def hilbert_phase(pd_series):

    return _np.unwrap(_np.angle(hilbert(pd_series)))


def iintegrate(series, initial=0):

    """
    Indefinite numeric integration of a pandas Series.

    Internally, the function calls scipy's 'cumtrapz' to integrate using
    the trapezoid rule. The index values are passed to cumtrapz as the
    axis to integrate over. The input series is also sorted prior to
    integration.

    Accepts a pandas Series, returns a pandas Series.

    Parameters
    ----------
    initial : value, default 0
        Sets the start value for indefinite integration
    """

    conditioned, nanmask = clean_series(series.sort_index())

    x_data = conditioned.index.values.astype(float)
    y_data = conditioned.values.astype(float)

    integrated = _spi.cumtrapz(y_data, x_data, initial)
    integrated = _np.insert(integrated, 0, initial)
    output = _pd.Series(integrated, index=series.sort_index().index)
    output[nanmask] = _np.nan

    return output


def dintegrate(series, xmin=None, xmax=None, closed=True):

    """
    Definite integration of a pandas Series.

    Calculates the definite integral of the input pandas series between
    the index values 'xmin' and 'xmax'. The integration interval can either
    be closed or open.

    Accepts a pandas Series, returns a value.

    Parameters
    ----------
    xmin : value, lower integration limit
    xmax : value, upper integration limit
    closed : boolean, default True
        Sets whether or not the integration interval includes the endpoints
        'xmin' and 'xmax'
    """

    raw = clean_series(_gp.strip_ends_from(series.sort_index()))[0]
    x_data = raw.index.values.astype(float)

    if xmin is None:
        xmin = x_data[0]
    if xmax is None:
        xmax = x_data[-1]

    if closed is True:
        sliced = raw[(x_data >= xmin) & (x_data <= xmax)]
    else:
        sliced = raw[(x_data > xmin) & (x_data < xmax)]

    return _spi.trapz(sliced.values.astype(float),
                      sliced.index.values.astype(float))


def align_series(series_a, series_b):

    ccorr = correlate(series_a, series_b, as_series=True)
    offset = _np.min(ccorr.where(ccorr == ccorr.max()).dropna().index.values)

    new_b = series_b.copy().shift(-offset)

    return new_b, offset


def align_df(pd_dframe, reference=None, series_list=None):

    df = pd_dframe.copy()
    offset_list = []

    if series_list is None:
        names = pd_dframe.columns.values
    else:
        names = _np.array(series_list)

    if reference is not None:
        mask = _np.array([reference == name for name in names])
        series_list = names[~mask]

        for loop_col in series_list:

            new_col, offset = align_series(df[reference], df[loop_col])
            df[loop_col] = new_col
            offset_list.append(offset)

        return df, offset_list

    else:
        sta = names

        for index, loop_col in enumerate(sta[1:]):

            new_col, offset = align_series(df[sta[index]], df[loop_col])
            df[loop_col] = new_col
            offset_list.append(offset)

        return df, offset_list


class FitFunction():

    def __init__(self, fitfunc):
        self.fitfunc = fitfunc

    def __call__(self, x_data, p):
        return self.fitfunc(_np.array(x_data), *p)

    def fit(self, p0, series, with_data=True):
        fitp = _spo.curve_fit(self.fitfunc, p0, series)[0]
        return FitObject(self.fitfunc, fitp, series, with_data)


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

    def limited(x, p):
        p = _np.array(p)
        output = _gp.iterfy(func(x, p))

        scale = _np.median(output) / len(output)

        lower = pmin - p
        upper = p - pmax

        lower[lower < 0] = 0
        upper[upper < 0] = 0

        output = output * scale * (_np.prod(_np.exp(lower)) +
                                   _np.prod(_np.exp(upper)))

    return limited


def _general_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def _weighted_general_function(params, xdata, ydata, function, weights):
    return weights * (function(xdata, *params) - ydata)


def curve_fit(f, pd_series, p0=None, sigma=None, **kw):
    """
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

    if _np.isscalar(p0):
        p0 = _np.array([p0])

    args = (xdata, ydata, f)
    if sigma is None:
        func = _general_function
    else:
        func = _weighted_general_function
        args += (1.0/_np.asarray(sigma),)

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
