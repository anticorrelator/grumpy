import numpy as _np
import grumpy as _gp
import pandas as _pd
import bottleneck as _bn
import scipy.stats as _sps
import pickle as _p


def join_bscans(file_iter):
    file_iter = _gp.iterfy(file_iter)
    df_iter = [_pd.read_table(dfile, index_col='time') for dfile in file_iter]

    ramp_count = [len(df.ramp_index.unique()) for df in df_iter[:-1]]
    cumulative_counts = _np.cumsum(ramp_count)

    for index, df in enumerate(df_iter[1:]):
        df.ramp_index += cumulative_counts[index]

    return RawBScan(_pd.concat(df_iter))


class RawBScan():
    def __init__(self, data, cl_object=None):

        self.data = data
        self.temp = data.thermometer_c.median()

        if cl_object is None:
            self.cantilever = _gp.Bunch()
            self.cantilever.w = 300e-6
            self.cantilever.l = 500e-6
            self.cantilever.t = 120e-9
            self.cantilever.f0 = data.cantilever_frequency.median()
            self.cantilever.angle = 45
            self.cantilever.mode = 1
            self.cantilever.density = 2.33e3

            self.cantilever.ring = _gp.Bunch()
            self.cantilever.ring.spacing = 800e-9
            self.cantilever.ring.linewidth = 50e-9
            self.cantilever.ring.thickness = 50e-9
            self.cantilever.ring.dia = 250e-9
            self.cantilever.ring.density = 19.3e3
            self.cantilever.ring.fraction = .4
            self.cantilever.ring.number = self._total_rings()

            self.cantilever.k = self._cantilever_k()

        else:
            self.cantilever = cl_object

    def _total_rings(self):
        pad_area = self.cantilever.w * \
            self.cantilever.ring.fraction * self.cantilever.l
        ring_area = self.cantilever.ring.spacing ** 2
        return pad_area / ring_area

    def _cantilever_k(self):
        # au_mass = self.cantilever.ring.number * self.cantilever.ring.density \
        #     * self.cantilever.ring.thickness * self.cantilever.ring.dia * 4 \
        #     * self.cantilever.ring.linewidth

        cantilever_mass = self.cantilever.w * self.cantilever.l * \
            self.cantilever.t * self.cantilever.density
        cantilever_emass = (33 / 140) * cantilever_mass

        # total_mass = cantilever_emass + au_mass
        total_mass = cantilever_emass
        return (2 * _np.pi * self.cantilever.f0) ** 2 * total_mass

    def _ab_range(self, dia=None, w=None, t=None, angle=None):

        if dia is None:
            dia = 250e-9
        if w is None:
            w = 50e-9
        if t is None:
            t = 50e-9
        if angle is None:
            angle = 45

        theta = angle * _np.pi / 180
        dia0 = _np.cos(theta) * dia
        delta = w * _np.cos(theta) + t * _np.sin(theta)
        phi0 = 6.62607e-34 / 1.60218e-19
        return [(dia0 - delta) * (dia - w) * _np.cos(theta) / phi0,
                (dia0 + delta) * (dia + w) * _np.cos(theta) / phi0]

    def _lowess_window(self, abperiod=None, window=5, **kwargs):

        if abperiod is None:
            abperiod = _gp.g_mean(self._ab_range(**kwargs)) ** -1

        b = self.raw.index.values
        span = max(b) - min(b)
        frac = (window * abperiod) / span

        if frac > 1:
            return 1
        else:
            return frac

    def smooth_with_lowess(self, abperiod=None, window=5, step=.005, **kwargs):

        if abperiod is None:
            abperiod = _gp.g_mean(self._ab_range()) ** -1

        ramps = self.data.ramp_index.unique()
        background = []

        for ramp in ramps:
            temp = self.data[self.data.ramp_index == ramp]
            ab_count = (temp.b_field.max() - temp.b_field.min()) / abperiod
            frac = window / ab_count
            if frac > 1:
                frac = 1
            background.append(_gp.lowess(temp.cantilever_frequency,
                              frac=frac))

        background = _np.hstack(background)
        return SmoothedBScan(self, background, step, **kwargs)


class SmoothedBScan(RawBScan):
    def __init__(self, oldbscan, background, step, **kwargs):
        super().__init__(oldbscan.data, oldbscan.cantilever)
        self.data['smoothed'] = self.data.cantilever_frequency - background
        self.data['background'] = background
        self._b = self.data.b_field.interpolate(method='slinear')
        self._step = step

        self.bin(step, **kwargs)

    def _sensitivity_factor(self):
        angle = self.cantilever.angle
        nrings = self.cantilever.ring.number
        aring = self.cantilever.ring.dia ** 2
        f0 = self.cantilever.f0
        k = self.cantilever.k
        L = self.cantilever.l

        alpha = 1.377

        theta = angle * _np.pi / 180
        return - _np.sqrt(nrings) * (f0 / (2 * k)) * \
            ((alpha / L) ** 2) * aring * (_np.cos(theta) ** 2 /
                                          _np.sin(theta))

    def _pc_integration_factor(self, harmonic=1):
        b_freq = [harmonic * x for x in self._ab_range()]
        return (2 * _np.pi * _gp.g_mean(b_freq)) ** -1

    def _to_current(self, df, harmonic=1, sensitivity=None):

        if sensitivity is None:
            sensitivity = self._sensitivity_factor()

        b = df.index.values.astype(float)

        self.di = df.apply(lambda x: x / (sensitivity * b ** 2))
        self.pc = self.di * self._pc_integration_factor(harmonic=harmonic)
        self.ab = self.pc.apply(_bn.nanmean, axis=1)

    def _bootstrap(self, data, blocks):
        count = len(data)
        data = _gp.iterfy(data)
        bootstrapped = _gp.block_bootstrap(data, _np.floor(count / blocks))
        return _bn.nanstd([_bn.nanmean(x) for x in bootstrapped])

    def _calculate_scatter(self, window=3):
        ramps = self.df.columns
        std_dict = {c: [_bn.nanstd(self.df[c][window*i: window*(i+1)])
                    for i in range(len(self.df[c]))] for c in ramps}
        std = _pd.DataFrame(std_dict)
        std_df = _pd.DataFrame({c: [std[c][_np.floor(i / window)] for i in
                               range(len(self.df[c]))] for c in ramps})
        return std_df

    def _block_b(self, step):
        return step * _np.floor(self._b / step) + (step / 2)

    def bin(self, step, harmonic=1, std_window=3, **kwargs):
        self.data['b_field'] = self._block_b(step)
        bunched = self.data.groupby(['b_field', 'ramp_index'])

        self.df = bunched.smoothed.apply(_gp.robust_mean, **kwargs).unstack()
        self.std = self._calculate_scatter(std_window)
        self.thermometer_c = bunched.thermometer_c.apply(_bn.nanmean).unstack()
        self.thermometer_d = bunched.thermometer_d.apply(_bn.nanmean).unstack()
        self.lockin_r = bunched.lockin_r.apply(_bn.nanmean).unstack()
        self._to_current(self.df, harmonic=harmonic)

    def bin_with(self, step, method, harmonic=1, **kwargs):
        self.data['b_field'] = step * _np.floor(self._b / step) + (step / 2)
        bunched = self.data.groupby(['b_field', 'ramp_index'])

        self.df = bunched.smoothed.apply(method, **kwargs).unstack()
        self.std = bunched.smoothed.apply(_sps.nanstd).unstack()
        self.thermometer_c = bunched.thermometer_c.apply(_bn.nanmean).unstack()
        self.thermometer_d = bunched.thermometer_d.apply(_bn.nanmean).unstack()
        self.lockin_r = bunched.lockin_r.apply(_bn.nanmean).unstack()
        self._to_current(self.df, harmonic=harmonic)

    def drop_ramps(self, ramps_to_drop):
        ramps_to_drop = _gp.iterfy(ramps_to_drop)
        ramp_list = self.df.columns

        ramp_mask = _np.array([any(_np.array(ramps_to_drop) == ramp)
                              for ramp in ramp_list])
        ramp_list = ramp_list[~ramp_mask]

        self.df = self.df.filter(items=ramp_list)
        self.std = self.std.filter(items=ramp_list)
        self.thermometer_c = self.thermometer_c.filter(items=ramp_list)
        self.thermometer_d = self.thermometer_d.filter(items=ramp_list)
        self.lockin_r = self.lockin_r.filter(items=ramp_list)
        self._to_current(self.df)

    def drop_data(self, column, bmin=None, bmax=None):

        if bmin is None:
            bmin = -_np.inf
        if bmax is None:
            bmax = _np.inf

        excl = (self.df.index > bmin) & (self.df.index < bmax)

        self.df[column] = self.df[column].where(~excl)
        self.std[column] = self.std[column].where(~excl)
        self.thermometer_c[column] = self.thermometer_c[column].where(~excl)
        self.thermometer_d[column] = self.thermometer_d[column].where(~excl)
        self.lockin_r[column] = self.lockin_r[column].where(~excl)
        self._to_current(self.df)

    def psd(self):
        return _gp.psd(self.ab)

    def fft(self):
        return _gp.absfft(self.ab)

    def save(self, targetfile):
        _pd.to_pickle(BScan(self), targetfile)


class BScan():
    def __init__(self, smoothed):

        self.df = smoothed.df
        self.di = smoothed.di

        smoothed._to_current(smoothed.df, harmonic=1)
        self.pc = smoothed.pc
        self.ab = smoothed.ab

        smoothed._to_current(smoothed.df, harmonic=2)
        self.pc2 = smoothed.pc
        self.ab2 = smoothed.ab

        self.std = smoothed.std
        self.temp = smoothed.temp
        self.ab_range = smoothed._ab_range()
        self.ab_range2 = [2 * x for x in self.ab_range]

        self.thermometer_c = smoothed.thermometer_c
        self.thermometer_d = smoothed.thermometer_d
        self.lockin_r = smoothed.lockin_r
        self.cantilever = smoothed.cantilever

    def psd(self):
        return _gp.psd(self.ab)

    def fft(self):
        return _gp.absfft(self.ab)

    def plot_psd(self):
        ax = self.psd().plot()
        r1 = self.ab_range
        c1 = _gp.g_mean(r1)
        s1 = r1[1] - r1[0]
        r2 = self.ab_range2
        s2 = r2[1] - r2[0]
        c2 = _gp.g_mean(r2)

        ax.axvspan(r1[0], r1[1], color='pink', alpha=.3)
        ax.axvspan(r2[0], r2[1], color='yellow', alpha=.3)
        ax.axvspan(c1 - .2 * s1, c1 + .2 * s1, color='red', alpha=.2)
        ax.axvspan(c2 - .2 * s2, c2 + .2 * s2, color='orange', alpha=.2)
        return ax

    def plot_fft(self):
        ax = self.fft().plot()
        r1 = self.ab_range
        c1 = _gp.g_mean(r1)
        s1 = r1[1] - r1[0]
        r2 = self.ab_range2
        s2 = r2[1] - r2[0]
        c2 = _gp.g_mean(r2)

        ax.axvspan(r1[0], r1[1], color='pink', alpha=.3)
        ax.axvspan(r2[0], r2[1], color='yellow', alpha=.3)
        ax.axvspan(c1 - .2 * s1, c1 + .2 * s1, color='red', alpha=.2)
        ax.axvspan(c2 - .2 * s2, c2 + .2 * s2, color='orange', alpha=.2)
        return ax
