import numpy as _np
import grumpy as _gp


def jackknife(pd_series, block_size=1):
    raw = _gp.clean_series(pd_series)[0].values.astype(float)
    blocks = _np.floor(_np.arange(len(raw)) / block_size)
    jacked = [raw[blocks != x] for x in _np.unique(blocks)]
    return _np.array(jacked)


def bootstrap(pd_series, block_size, samples):

    """
    BOOTSTRAP randomly selects random subsets from the input pandas Series
    "pd_series". Each random subset has length "block_size". Internally calls
    numpy.random.choice.
    """

    from numpy.random import choice as _rc
    cleaned = _gp.clean_series(pd_series)[0]
    raw = cleaned.values.astype(float).tolist()
    samplings = _np.array([_rc(raw, block_size) for count in range(samples)])

    return samplings


def block_bootstrap(pd_series, size, blocks=None):

    """
    BLOCK_BOOTSTRAP creates subsamples of data specified by "blocks" of length
    "size". Statistical properties should only differ from true bootstrapping
    if there are point-to-point correlations in the data.
    """

    cleaned = _gp.clean_series(pd_series)[0]
    raw = cleaned.values.astype(float)

    if blocks is None:
        blocks = _np.floor(len(raw) / size)
        samplings = _np.reshape(raw[:(blocks * size)], (blocks, size))

    elif blocks is 'all':
        samplings = _np.array([raw[stpt:(stpt + size)]
                              for stpt in range(len(raw) - size)])

    elif type(blocks) is int:
        from random import sample as _rs
        starts = _rs(range(len(raw) - size), blocks)
        samplings = _np.array([raw[stpt:(stpt + size)]
                              for stpt in starts])

    return samplings
