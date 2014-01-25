import numpy as _np

_hbar = 1.05457e-34
_kb = 1.38065e-23
_ee = 1.60218e-19


def _sumterm(x, order):
    if order == 1:
        return _np.exp(- _np.sqrt(2 * _np.pi ** 3 * x))
    return order * _np.exp(- _np.sqrt(2 * _np.pi ** 3 * order * x)) + \
        _sumterm(x, order-1)


def _tscaling(x, order):
    return _np.sqrt(_np.pi ** 6 / 3 * x ** 2 * _sumterm(x, order))


def _tscale(diffc, lring):
    return _hbar * _np.pi ** 2 * diffc / (_kb * lring ** 2)


def _pc0(p, diffc, lring):
    return .37 * (p ** -1.5) * 3 * _ee * diffc / (lring ** 2)


def pc_theory(p, diffc, lring, temp, order=100):
    return _pc0(p, diffc, lring) * _tscaling(p ** 2 * temp /
                                           _tscale(diffc, lring), order)
