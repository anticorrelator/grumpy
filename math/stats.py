import numpy as _np
import grumpy as _gp


def bootstrap(pd_series, block_size, samples):

    """
    BOOTSTRAP randomly selects random subsets from the input pandas Series
    "pd_series". Each random subset has length "block_size". Internally calls
    random.sample.
    """

    from random import sample as _rs
    cleaned = _gp.clean_series(pd_series)[0]
    raw = cleaned.values.astype(float).tolist()
    samplings = _np.array([_rs(raw, block_size) for count in range(samples)])

    return samplings


def block_bootstrap(pd_series, size, blocks=None):

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
