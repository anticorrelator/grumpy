import numpy as _np
import pandas as _pd
import cylowess as _cl
import scipy as _sp
import scipy.stats as _sps
import scipy.fftpack as _fft
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
    """

    nanmask = _np.isnan(pd_series.values.astype(float))
    clean_data = pd_series.interpolate().fillna(method='bfill')
    return clean_data, nanmask


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
            to perform.
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


def pdcorr(series_a, series_b, as_series=False):
    clean_a = clean_series(series_a)[0]
    clean_b = clean_series(series_b)[0]
    correlated = _sp.correlate(clean_a, clean_b, 'same')

    if as_series is False:
        return correlated.values.astype(float)
    else:
        indices = _np.arange(_np.max((len(series_a), len(series_b))))
        center = _np.ceil(_np.median(indices))
        offset = indices - center
        return _pd.Series(correlated.values.astype(float), index=offset)


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

    output = _np.empty(len(series))
    output[:] = _np.nan

    x_data = series.dropna().index.values.astype(float)
    y_data = series.dropna().values.astype(float)

    fft_length = _np.ceil(len(series) / 2)
    fft_spacing = _np.mean(_np.diff(x_data))
    fft_range = max(x_data) - min(x_data)

    fft_freq = _fft.fftfreq(len(series), d=fft_spacing)
    fft_mag = _fft.fft(y_data)
    fft_mag[-fft_length:] = _np.nan

    output[0:len(fft_mag)] = fft_mag

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

    nanmask = _np.isnan(series.sort_index().values.astype(float))
    conditioned = series.sort_index().fillna(0)

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

    raw = series.sort_index()
    x_data = raw.index.values.astype(float)

    if xmin is None:
        xmin = x_data[0]
    if xmax is None:
        xmin = x_data[-1]

    if closed is True:
        sliced = raw[(x_data >= xmin) & (x_data <= xmax)]
    else:
        sliced = raw[(x_data > xmin) & (x_data < xmax)]

    return _spi.trapz(sliced.values.astype(float),
                      sliced.index.values.astype(float))


def align_series(series_a, series_b):

    ccorr = pdcorr(series_a, series_b, as_series=True)
    offset = _np.min(ccorr.where(ccorr == ccorr.max()).dropna().index.values)

    new_b = series_b.copy().shift(offset)

    return new_b, offset


def align_df(pd_dframe, ref_col=None, series_to_align=None):

    df = pd_dframe.copy()
    offset_list = []

    if series_to_align is None:
        names = pd_dframe.columns.values
    else:
        names = _np.array(series_to_align)

    if ref_col is not None:
        mask = _np.array([ref_col == name for name in names])
        series_to_align = names[~mask]

        for loop_col in series_to_align:

            new_col, offset = align_series(df[ref_col], df[loop_col])
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

    def __call__(self, p, x_data):
        return self.fitfunc(p, _np.array(x_data))

    def fit(self, p0, series, with_data=True):
        fitp = lsfit(self.fitfunc, p0, series)[0]
        return FitObject(self.fitfunc, fitp, series, with_data)


class FitObject():

    def __init__(self, fitfunc, fitp, series, with_data=True):
        self.fitfunc = lambda x: fitfunc(fitp, x)
        self.fitp = fitp

        if with_data is True:
            self.data = _pd.DataFrame(series, columns=['Raw'])
            self.data['Fitted Curve'] = self(series.index.values.
                                             astype(float))
            self.data['Residuals'] = _np.abs(self.data['Raw'] -
                                             self.data['Fitted Curve'])

    def __call__(self, x_data):
        return self.fitfunc(x_data)


def lsfit(fitfunc, p0, series, **kwargs):

    x_data = series.index.values.astype(float)
    y_data = series.values.astype(float)

    errfunc = lambda p, x, y: fitfunc(p, x) - _np.ravel(y)

    return _spo.leastsq(errfunc, p0, args=(x_data, y_data), **kwargs)
