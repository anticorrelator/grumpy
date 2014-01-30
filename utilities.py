import numpy as _np
import pandas as _pd


class Bunch:
    """
    Bunch is a utility class used to group named attributes into a single
    object. Bunch can be called with keyword arguments to set these
    attributes or they can be assigned after instantiation.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    pass


def iterfy(x):
    """
    Checks to see if the input is iterable. If not, the input is nested in a
    list so that it is. Useful for sanitizing inputs to functions that get
    looped over.
    """
    if isinstance(x, str):
        x = [x]
    try:
        iter(x)
    except TypeError:
        x = [x]
    return x


def strip_ends_from(pd_series, value_to_strip=None):

    if type(pd_series) is _pd.Series:
        svalues = pd_series.values.astype(float)
    else:
        svalues = _np.array(pd_series)

    if value_to_strip is None:
        bool_values = _np.isnan(svalues)
    else:
        bool_values = _np.array(svalues == value_to_strip)

    left_mask = []
    for bools in bool_values:
        if len(left_mask) is 0:
            left_mask.append(bools)
        else:
            left_mask.append(bools and left_mask[-1])

    right_mask = []
    for bools in bool_values[::-1]:
        if len(right_mask) is 0:
            right_mask.append(bools)
        else:
            right_mask.append(bools and right_mask[-1])

    right_mask = right_mask[::-1]
    strip_mask = [l or r for l, r in zip(left_mask, right_mask)]
    strip_mask = _np.array(strip_mask)

    return pd_series[~strip_mask]
