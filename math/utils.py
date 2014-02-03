import numpy as _np
import pandas as _pd


def strip_ends_from(pd_series, value_to_strip=None):

    if type(pd_series) is _pd.Series:
        svalues = pd_series.values.astype(float)
    else:
        svalues = _np.asarray(pd_series)
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
    strip_mask = _np.asarray([l or r for l, r in zip(left_mask, right_mask)])

    return pd_series[~strip_mask]
