import numpy as _np
import statsmodels as _sm


def reject_outliers(data, deviation_tolerance=3):

    """
    Replaces outliers in a numpy array with NaN.

    Checks for outliers by comparing the distance of each element in the
    input array from the median value of the input array. The median is
    used to be robust to large outliers.

    Accepts numpy array, returns numpy array of the same length.

    Parameters
    ----------
    deviation_tolerance : value, default 3
        deviation_tolerance sets the threshold for whether an element from
        the input vector is removed by the outlier rejection.
    """

    data = data.astype('float')
    distance = _np.abs(data - _np.median(data))
    data[distance > deviation_tolerance*_np.median(distance)] = _np.nan
    return data


def robust_mean(data, stdcutoff=None, deviation_tolerance=3):

    """
    Robustified mean. Rejects outliers before taking mean.

    Significant speedup when utilizing the "stdcutoff" argument.

    Accepts numpy array, returns value.

    Parameters
    ----------
    stdcutoff : value, default None
        stdcutoff is compared to the std. dev. of input data. Explicit
        outlier rejection only occurs if this test fails. 10x+ speedup.
    deviation_tolerance : value, default 3
        deviation_tolerance sets the threshold for whether an element from
        the input vector is removed by the outlier rejection.
    """

    if stdcutoff is None:
        return _np.mean(reject_outliers(data,
                        deviation_tolerance=deviation_tolerance))
    else:
        if _np.std(data) < stdcutoff:
            return _np.mean(data)
        else:
            return _np.mean(reject_outliers(data,
                            deviation_tolerance=deviation_tolerance))


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
    nanmask = _np.isfinite(series).values
    x_data = series.dropna().index.values.astype(float)
    y_data = series.dropna().values.astype(float)
    fit_polynomial = _np.polyfit(x_data, y_data, order)
    output[nanmask] = _np.polyval(fit_polynomial, x_data)
    return output


def lowess(series, frac=.5):

    """
    Smooths a pandas Series using local linear regression (lowess).

    Outputs the fitted polynomial evaluated at the index values where the
    input series value is finite. NaN everywhere else.

    Accepts (single index) pandas Series, returns a numpy array of the same
    length.

    Parameters
    ----------
    frac : value between 0 and 1, default .5
        Fraction of data to span with filter window
    """

    output = _np.empty(len(series))
    output[:] = _np.nan
    nanmask = _np.isfinite(series).values
    x_data = series.dropna().index.values.astype(float)
    y_data = series.dropna().values.astype(float)

    smooth = _sm.nonparametric.lowess(y_data, x_data, frac=frac)
    output[nanmask] = smooth[:, 1]
    return output
