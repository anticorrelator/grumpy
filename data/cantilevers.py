import numpy as _np
import scipy as _sp
import pandas as _pd


def mode_gamma(mode_number):

    modes = [1.8751, 4.6941, 7.8548, 10.9955, 14.1372, 17.2788, 20.4204,
             23.5619, 26.7035, 29.8451, 32.9867, 36.1283, 39.2699, 42.4115,
             45.5531, 48.6947, 51.8363, 54.9779, 58.1195, 61.2611]

    return modes[mode_number - 1]


def flexf(cantilever_length, cantilever_width, cantilever_thickness,
          mode_number=1, e_modulus=169e9, density=2330):

    area_moment = cantilever_width * (cantilever_thickness ** 3 / 3)
    cross_section = cantilever_thickness * cantilever_width

    return _np.sqrt(((e_modulus * area_moment) / (4 * _np.pi ** 2 * density *
                    cross_section * cantilever_length ** 4)) *
                    mode_gamma(mode_number) ** 4)


def torsionf(cantilever_length, cantilever_width, cantilever_thickness,
             mode_number=1, shear_modulus=50.9e9, density=2330):

    polar_moment = (cantilever_width * cantilever_thickness ** 3 +
                    cantilever_width ** 3 * cantilever_thickness) / 12

    xi = 1 / 3 * cantilever_thickness ** 4 * (cantilever_width /
                                              cantilever_thickness - .630)

    return (2 * mode_number - 1) / (4 * cantilever_length) * \
        _np.sqrt((shear_modulus * xi) / (density * polar_moment))


class ringdown():

    def __init__(self, datafile):
        raw = _pd.read_table(datafile, header=None)
        raw['time'] = raw.ix[0, 2] * raw.index.values
        raw.set_index(['time'], inplace=True)

        self.f0 = raw.ix[0, 3]
        self.tc_guess = raw.ix[0, 4]
        self.data = raw.drop([2, 3, 4], 1)
        self.fit()

    def ringdownfunc(self, p0):
        return lambda time: p0[0] * _np.exp(time / p0[1])

    def fit(self):

        p0 = (self.data[0].max(), self.tc_guess)

        fitfunc = lambda p, time: p[0] * _np.exp(time / p[1])
        errfunc = lambda p, time, data: fitfunc(p, time) - _np.ravel(data)

        self.fitp = _sp.optimize.leastsq(errfunc, p0, args=(self.data.index.
                                         values.astype(float),
                                         self.data[0].values.astype(float)))
        self.q = _np.abs(_np.pi * self.fitp[0][1] * self.f0)

        return self
