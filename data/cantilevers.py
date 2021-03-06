import numpy as _np
import grumpy as _gp
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.special as _sps
import scipy.optimize as _spo


def mode_gamma(mode_number):

    modes = [1.8751, 4.6941, 7.8548, 10.9955, 14.1372, 17.2788, 20.4204,
             23.5619, 26.7035, 29.8451, 32.9867, 36.1283, 39.2699, 42.4115,
             45.5531, 48.6947, 51.8363, 54.9779, 58.1195, 61.2611]

    return modes[mode_number - 1]


def flexf(cantilever_length, cantilever_width, cantilever_thickness,
          mode_number=1, e_modulus=169e9, density=2330):

    area_moment = cantilever_width * (cantilever_thickness ** 3 / 12)
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


class Ringdown():

    def __init__(self, datafile):
        raw = _pd.read_table(datafile, header=None)
        raw['time'] = raw.ix[0, 2] * raw.index.values
        raw.set_index(['time'], inplace=True)
        self.f0 = raw.ix[0, 3]
        self.data = raw.drop([2, 3, 4], 1)
        self.ringdown = self.data[0]
        self.initial_guess = [self.data[0].max(), raw.ix[0, 4]]
        self.fit()


    def ringdownfit(self, time):
        return self.amplitude * _np.exp(-time / self.tau)


    def fit(self, e_folds=3, **kwargs):
        logdata = _np.log(self._truncated_ringdown(e_folds))
        p0 = self.initial_guess
        self.fitp = _np.polyfit(logdata.index.values, logdata.values, 1)
        self.amplitude = _np.exp(self.fitp[1])
        self.tau = -1 / self.fitp[0]
        self.q = _np.abs(_np.pi * (self.fitp[0] ** -1) * self.f0)
        return self


    def plot(self):
        tpoints = self.data.index.values.astype(float)
        f, ax = _plt.subplots()
        ax.plot(tpoints, self.data[0].values.astype(float), alpha=.2)
        ax.plot(tpoints, self.ringdownfit(tpoints), lw=3, alpha=.8)
        f.suptitle('Cantilever ringdown')
        ax.set_ylabel('Lock-in amplitude [V]')
        ax.set_xlabel('Time [seconds]')


    def _truncated_ringdown(self, e_folds):
        e_fold_factor = 1 / (e_folds * _np.e)
        ringdown = self.ringdown
        maximum = ringdown.max()
        truncation = ringdown.ix[ringdown < maximum * e_fold_factor].index[0]
        return ringdown.ix[:truncation]


class Calibration():

    def __init__(self, ampscanfile, fiber_position, mode_number):

        params = _pd.read_table(ampscanfile, skiprows=1, nrows=1, header=None,
                                index_col=0)
        self.f0 = params[1]
        self.wavelength = 1550e-9
        self.fiber_position = fiber_position
        self.mode_number = mode_number

        self.data = _pd.read_table(ampscanfile, skiprows=3, header=None,
                                   index_col=0)
        self.data[2].mul(1e-3)

        self.fit()

    def _calibrate(self):
        vd_max = 1.842 / self.fitp[1]
        self.x2v = 4 * _np.pi * vd_max / (1.842 * self.wavelength)
        return self


    def peak_drive(self, drive):
        # extra_factors accounts for the Vpeak-Vrms conversion as well as
        # compensating for the fiber position not being at the cantilever tip
        extra_factors = self.modeshape(self.fiber_position, self.mode_number)
        return drive / self.x2v / extra_factors

    def besselfunc(self, p=None):

        if p is None:
            return lambda p0, p1, x: p0 * _np.abs(_sps.jn(1, p1 * x))
        else:
            return lambda x: p[0] * _np.abs(_sps.jn(1, p[1] * x))
    def modeshape(self, fiber_position, nmode):

        alpha = [-1.3622, -.9819, -1.008, -1.000]
        a = alpha[nmode - 1]

        beta = [1.875, 4.694, 7.855, 10.996]
        b = beta[nmode - 1]

        shape = lambda z: a * (_np.cos(b * z) - _np.cosh(b * z)) + \
            (_np.sin(b * z) - _np.sinh(b * z))

        return shape(fiber_position) / shape(1)

    def p0(self):
        data = self.data[2]
        argmax = data.argmax()
        return (data.max() / .5819, 1.842 / argmax, 0)

    def fit(self):
        data = self.data[2]
        argmax = data.argmax()
        data = data[:(argmax * 1.2)]
        # fitfunc = self.besselfunc()
        # self._fitoutput = _gp.curve_fit(data[:argmax], fitfunc, self.p0())
        # self.fitp = self._fitoutput.fitp
        self.fitp = self.p0()
        self.besselfit = self.besselfunc(self.fitp)
        self._calibrate()

        return self


# class Cantilever(Ringdown, Calibration):
