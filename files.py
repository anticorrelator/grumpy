import pandas as _pd


def pload(filename):
    return _pd.read_pickle(filename)
