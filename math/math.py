import numpy as _np
import grumpy as _gp
import pandas as _pd
import cylowess as _cl
import scipy as _sp
import bottleneck as _bn
import scipy.fftpack as _fft
import scipy.integrate as _spi
import functools as _functools
from itertools import product as _product


def dataframer(func):
    """
    DATAFRAMER is a decorator that accepts functions that take pandas Series
    as their first input. If the input to those functions is a DataFrame,
    it broadcasts the input function "func" over each column or row as a
    Series.

    If the input is a ndarray-like object with dimension <=2, DATAFRAMER will
    attempt to cast the input as a DataFrame. Optional "index" and "columns"
    parameters can be passed into the input function to set index or column
    values to be used during the Series or DataFrame construction.

    DATAFRAMER is expensive. Native methods and functions that are broadcast
    internally are much faster. Use DataFramer as a convenience wrapper or if
    broadcasting over DataFrames is not easily done.

    Can be invoked using the @dataframer syntax before a function definition.

    Parameters
    ----------
    func : callable

    Returns
    -------
    broadcasted : callable
    """
    def broadcasted(*args, axis=0, index=None, columns=None, **kw):

        raw = args[0]
        params = args[1:]
        allowed = (_np.ndarray, _pd.Series, _pd.DataFrame)
        if not isinstance(args[0], allowed):
            type_msg = 'Input must be array-like.'
            raise TypeError(type_msg)

        if isinstance(args[0], _pd.Series):
            return func(*args, **kw)

        if not isinstance(args[0], _pd.DataFrame):
            if raw.ndim > 2:
                dim_msg = 'Array has too many dimensions to be cast \
                    into a DataFrame'
                raise TypeError(dim_msg)
            if raw.ndim == 1:
                df = _pd.Series(raw, index=index)
                return func(df, *params, **kw)
            if axis > 1:
                dim_msg = 'DataFrames only have axis 1 or 0'
                raise TypeError(dim_msg)
            else:
                df = _pd.DataFrame(raw, index=index, columns=columns)
        else:
            df = raw

        if axis == 0:
            outs = {k: func(s, *params, **kw) for k, s in df.iteritems()}
            if all([_np.isscalar(v) for v in outs.values()]):
                return _pd.Series(outs)
            else:
                return _pd.DataFrame.from_dict(outs, orient='columns')

        elif axis == 1:
            outs = {k: func(s, *params, **kw) for k, s in df.iterrows()}
            return _pd.DataFrame.from_dict(outs, orient='index')

    return broadcasted


def axify(func=None, output='vector', axify_dim=1):
    """
    AXIFY is a decorator that broadcasts a function that accepts a
    1-dimensional numpy array to arbitrary dimensions. The decorated function
    also gains an "axis" argument which will apply the function along the
    specified axis.
    """
    if output not in ['vector', 'scalar']:
        msg = '"output" argument must be either "vector" or "scalar".'
        raise TypeError(msg)

    if func is None:
        return _functools.partial(axify, output=output, axify_dim=axify_dim)

    @_functools.wraps(func)
    def axified(*args, axis=None, **kwargs):
        if not isinstance(args[0], _np.ndarray):
            type_msg = 'Input must be array-like.'
            raise TypeError(type_msg)
        if args[0].ndim >= axify_dim:
            dat = args[0]
            dims = dat.shape
            if axis is None:
                axis = dat.ndim - 1
            if axis > dat.ndim:
                dim_msg = '"axis" out of range of input dimension'
                raise TypeError(dim_msg)

            if output is 'vector':
                output_array = _np.zeros(dims)
            elif output is 'scalar':
                outputdims = dims[:axis] + tuple([1]) + dims[axis+1:]
                output_array = _np.zeros(outputdims)

            slicer = tuple([slice(None)])
            for r in _product(*(range(x) for x in dims[:axis]+dims[axis+1:])):
                subindex = r[:axis] + slicer + r[axis:]
                subarray = dat[subindex].ravel()
                output_array[subindex] = func(subarray, *args[1:], **kwargs)

            return _np.squeeze(output_array)

        else:
            return func(*args, **kwargs)
    return axified


def quadrature(funcs):
    pass


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

    Parameters
    ----------
    kwargs : keywords
        Passed to pandas Series.interpolate method
    """

    nanmask = _np.isnan(pd_series.values.astype(float))

    if any(nanmask):
        clean_data = pd_series.interpolate(method='slinear')
        clean_data = clean_data.fillna(method='bfill')
    else:
        clean_data = pd_series

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

    data_length = len(pd_series.index)
    data_indices = pd_series.dropna().index.values.astype(float)
    data_points = pd_series.dropna().values.astype(float)

    small_index = _np.min(data_indices)
    big_index = _np.max(data_indices)
    forced_indices = _np.linspace(small_index, big_index, data_length)

    forced_data = _sp.interp(forced_indices, data_indices, data_points)

    return _pd.Series(forced_data, index=forced_indices)


def index_uniformity_of(pandas_obj):

    """
    Checks if the indices of a pandas Series or DataFrame is uniformly spaced.
    """

    data_indices = pandas_obj.index.values.astype(float)
    spacing = _np.diff(data_indices)
    if _np.allclose(spacing, [_bn.nanmean(spacing)] * len(spacing), 1e-4):
        return True
    else:
        return False


def demedian(data, axis=None):
    """
    Subtracts the median from a dataset.

    Parameters
    ----------
    data : array-like
        Input array. Must have dimension 2 or less.
    axis : int, optional, default (ndim - 1)
        applies DEMEDIAN along a specific axis. Must be either 0 or 1.

    Returns
    -------
    ds : numpy array of the same dimension as "data"
    """
    ds = data.astype(float).copy()

    if data.ndim > 2:
        dim_msg = 'Expected 2 or fewer dimensions'
        raise TypeError(dim_msg)

    if axis is None:
        axis = ds.ndim - 1

    if ds.ndim == 1:
        demedianed = ds - _bn.nanmedian(ds, axis=axis)
    elif axis == 0:
        demedianed = _np.array([x - _bn.nanmedian(x) for x in ds.T]).T
    elif axis == 1:
        demedianed = _np.array([x - _bn.nanmedian(x) for x in ds])

    return demedianed


def demean(data, axis=None):
    """
    Subtracts the mean from a dataset.

    Parameters
    ----------
    data : array-like
        Input array. Must have dimension 2 or less.
    axis : int, optional, default (ndim - 1)
        applies DEMEDIAN along a specific axis. Must be either 0 or 1.

    Returns
    -------
    ds : numpy array of the same dimension as "data"
    """
    ds = data.astype(float)

    if data.ndim > 2:
        dim_msg = 'Expected 2 or fewer dimensions'
        raise TypeError(dim_msg)

    if axis is None:
        axis = ds.ndim - 1

    if ds.ndim == 1:
        demeaned = ds - _bn.nanmean(ds, axis=axis)
    elif axis == 0:
        demeaned = _np.array([x - _bn.nanmean(x) for x in ds.T]).T
    elif axis == 1:
        demeaned = _np.array([x - _bn.nanmean(x) for x in ds])

    return demeaned


def trim(data, reject=.5, axis=None):

    """
    Replaces extremal values in a numpy array or list with NaN.

    Sorts data and removes largest and smallest values. Percentage of
    original dataset removed determined by "reject" parameter. TRIM will not
    remove all data if reject approaches 1. The default "reject" paramter .5
    returns values within the inter-quartile range (IQR).

    Parameters
    ----------
    data : array-like
        Input array. Must have dimension 2 or less.
    reject : value between 0 and 1, optional, default .5
        sets the percentage of data to be rejected. (reject=.5)
        rejection strength implies that the largest and smallest 25%
        of data is rejected.
    axis : int, optional, default (ndim - 1)
        applies TRIM along a specific axis. Must be either 0 or 1.

    Returns
    -------
    ds : numpy array of the same dimension as "data"
    """

    ds = _np.array(data.astype(float).copy())

    if axis is None:
        axis = 0

    if ds.ndim > 2:
        dim_msg = 'Expected 2 or fewer dimensions'
        raise TypeError(dim_msg)
    if axis > ds.ndim:
        ax_msg = 'axis out of bounds'
        raise ValueError(ax_msg)
    if (reject < 0) or (reject > 1):
        r_msg = 'reject parameter must be between 0 and 1'
        raise ValueError(r_msg)

    dlen = ds.shape[axis]
    thresh = _np.floor(dlen * reject / 2)

    if thresh == 0:
        thresh += 1
    if thresh == dlen / 2:
        thresh -= 1

    rank = _np.argsort(ds, axis=axis)

    if ds.ndim is 1:
        ds[rank[-thresh:]] = _np.nan
        ds[rank[:thresh]] = _np.nan
    elif axis == 0:
        for x, r in zip(ds.T, rank.T):
            x[r[-thresh:]] = _np.nan
            x[r[:thresh]] = _np.nan
    elif axis == 1:
        for x, r in zip(ds, rank):
            x[r[-thresh:]] = _np.nan
            x[r[:thresh]] = _np.nan
    return ds


def reject_outliers(data, reject=.5, axis=None):

    """
    Replaces outliers in a numpy array with NaN.

    Checks for outliers by comparing the distance of each element in the
    input array from the median value of the input array. For symmetric
    distributions, the default "reject" parameter returns the values within
    the inter-quartile range (IQR).

    Parameters
    ----------
    data : array-like
        Input data. Must be dimension 2 or less.
    reject : value between 0 and 1, optional, default .5
        "reject" sets the percentage of data to be rejected.
    axis : int, optional, default (ndim - 1)
        Applies REJECT_OUTLIERS along a specific axis. Must be either 0 or 1.

    Returns
    -------
    ds : numpy array of the same dimension as "data"
    """

    ds = _np.array(data.astype(float).copy())

    if axis is None:
        axis = ds.ndim - 1

    if ds.ndim > 2:
        dim_msg = 'Expected 2 or fewer dimensions'
        raise TypeError(dim_msg)
    if axis > ds.ndim:
        ax_msg = 'axis out of bounds'
        raise ValueError(ax_msg)
    if (reject < 0) or (reject > 1):
        r_msg = 'reject parameter must be between 0 and 1'
        raise ValueError(r_msg)

    dlen = _np.shape(ds)[axis]
    thresh = _np.floor(dlen * reject)

    if thresh == 0:
        thresh += 1
    if thresh == dlen:
        thresh -= 1

    rank = _np.argsort(_np.abs(demedian(ds, axis=axis)), axis=axis)

    if ds.ndim is 1:
        ds[rank[-thresh:]] = _np.nan
    elif axis == 0:
        for x, r in zip(ds.T, rank.T):
            x[r[-thresh:]] = _np.nan
    elif axis == 1:
        for x, r in zip(ds, rank):
            x[r[-thresh:]] = _np.nan
    return ds


def robust_mean(data, reject=.5, method='trim', stdcutoff=None, axis=None):

    """
    Robustified mean. Rejects outliers using "method" before taking mean.
    Ignores NaNs.

    ROBUST_MEAN will skip outlier rejection of std(data) exceeds the optional
    "stdcutoff" argument. This can significantly speed up loops that call
    ROBUST_MEAN multiple times.

    Parameters
    ----------
    data : array-like
    reject : float, optional, default .5
    method : string, optional, default 'trim'
    stdcutoff : float, optional, default None
    axis : int, optional, default (ndim - 1)
    """

    if method == "trim":
        rejected = trim(data, reject=reject, axis=axis)
    elif method == "reject_outliers":
        rejected = reject_outliers(data, reject=reject, axis=axis)

    if stdcutoff is None:
        return _bn.nanmean(rejected, axis=axis)
    else:
        if _np.std(data) < stdcutoff:
            return _bn.nanmean(data, axis=axis)
        else:
            return _bn.nanmean(rejected, axis=axis)


def g_mean(data, axis=None):

    data = _np.asarray(data).copy()
    if any(data <= 0):
        pos_msg = 'Geometric mean is only defined for positive arguments.'
        raise ValueError(pos_msg)

    if axis is None:
        numel = _np.prod(data.shape)
    else:
        numel = data.shape[axis]
    return _np.prod(data, axis=axis) ** (1 / numel)


@dataframer
def delinear(data):
    """
    DELINEAR removes a linear trend from the input data. If the input data is
    a pandas series, the index values will be taken as the x-values. Otherwise
    the input data will be taken to be evenly spaced.

    An offset is added to the output to compensate for the removed linear
    trend.
    """

    if type(data) is _pd.Series:
        xs = data.values.astype(float)
    else:
        xs = range(len(data))

    if len(data) <= 3:
        return data

    fit = _np.array([_np.polyfit(xs, data, 1)[0], 0])
    line = _np.polyval(fit, xs)
    offset = _np.sign(fit[0]) * (line.max() - line.min()) / 2

    return data - line + offset


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


def loess(series, frac=.5, it=None):
    from grumpy.math.loess import _loess
    output = _np.empty(series.shape)
    output[:] = _np.nan
    nanmask = _np.isnan(series.values.astype(float))
    x_data = series.dropna().index.values.astype(float)
    y_data = series.dropna().values.astype(float)

    if it is None:
        it = 3

    smooth = _loess(y_data, x_data, frac=frac, it=it)
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

    wdict = {'hanning': _np.hanning,
             'hamming': _np.hamming,
             'bartlett': _np.bartlett,
             'blackman': _np.blackman}

    if not window in wdict.keys():
        msg = 'Window is not "flat", "hanning", "hamming", "bartlett", \
            or "blackman"'
        raise ValueError(msg)

    nanmask = _np.isnan(series.values.astype(float))
    x_data = series.fillna(method='bfill').values.astype(float)

    if window == 'flat':
        w = _np.ones(window_length, 'd')
    else:
        w = wdict[window](window_length)

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

        return _pd.DataFrame(dict(zip(series_list, new_cols)))

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
    import scipy.signal as _spsig
    data = force_spacing(pd_series)
    return _spsig.hilbert(data)


def hilbert_amplitude(pd_series):

    return (hilbert(pd_series) + pd_series).apply(_np.abs)


def hilbert_phase(pd_series):

    return _np.unwrap(_np.angle(hilbert(pd_series)))


@dataframer
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


@dataframer
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

    raw = _gp.strip_ends_from(series.sort_index()).dropna()
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
