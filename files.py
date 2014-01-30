def pload(filename):
    import pickle as _p
    return _p.load(open(filename, 'rb'))
