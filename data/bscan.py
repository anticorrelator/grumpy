import numpy as _np
import grumpy as _gp
import pandas as _pd
import matplotlib.pyplot as _plt
import scipy.integrate as _spi


def add_ramp_index(dataframe, offset=0, index_on='b_field', file_index=0):

    """
    Indexes a dataframe column with a linearly ramped structure.

    "add_ramp_index" takes the second derivative of a specified dataframe
    column and looks for inflection points. Ramps are separated by these
    inflection points.

    Accepts a pandas DataFrame, returns a pandas DataFrame with an additional
    'ramp_index' column and 'file_index' column.

    Parameters
    ----------
    offset : integer, default 0
        Adds a constant offset to the ramp indices. Useful for joining
        datasets.
    index_on : string, default 'b_field'
        Indexes on specified column identifier.
    file_index : integer, default 0
        Adds another index that's useful when merging or joining dataframes.
    """

    dataframe['ramp_index'] = dataframe[index_on].fillna(method='pad')
    dataframe['ramp_index'] = dataframe.ramp_index.diff().fillna(0)\
        .apply(_np.sign)
    nonzero = dataframe.ramp_index != 0
    dataframe.ramp_index[nonzero] = dataframe.ramp_index[nonzero]\
        .diff().fillna(0).apply(_np.abs).apply(_np.sign)
    dataframe.ramp_index = dataframe.ramp_index.cumsum() + offset

    dataframe['file_index'] = file_index * _np.ones(len(dataframe
                                                    .ramp_index))

    return dataframe


def join_bscans(df_list, **kwargs):

    """
    Adds sequential ramp indices and concatenates bscans.

    Accepts a list of pandas DataFrames, returns a single pandas DataFrame
    with an additional 'ramp_index' column and 'file_index' column.

    Parameters
    ----------
    **kwargs passed to 'add_ramp_index'
        index_on : string, default 'b_field'
            Indexes on specified column identifier.
    """

    ramp_counter = 0
    file_counter = 0
    for dataframe in df_list:
        dataframe = add_ramp_index(dataframe,
                                   offset=ramp_counter,
                                   file_index=file_counter,
                                   **kwargs)
        ramp_counter += len(dataframe.ramp_index.unique())
        file_counter += 1

    return _pd.concat(df_list)


def reduce_bscan(data, b=None, f=None, t=None, temp=None,
                 robust=True, stdcutoff=1, cantilever_object=None,
                 **kwargs):

    """
    Averages bscan data at each field point.

    'reduce_bscan' first adds a ramp_index column to the input DataFrame
    or list of DataFrames.

    Internally, 'reduce_bscan' calls the pandas DataFrame 'groupby' method
    and creates buckets of points indexed by ramp number and b_field. This
    groups together all points taken per "field step."

    The resulting buckets are aggregated using either 'mean' or
    'robust_mean'. Robust averaging can be selected using the boolean
    'robust' flag.

    Accepts a pandas DataFrame or list of pandas DataFrames, returns
    a ReducedBScan object.

    Parameters
    ----------
    b : string, default 'b_field'
        Name of DataFrame column with b_field data
    f : string, default 'cantilever_frequency'
        Name of DataFrame column with frequency data
    t : string, default 'measurement_time'
        Name of DataFrame column with timestamps
    temp : string, default 'thermometer_c'
        Name of DataFrame column with temperature data
    robust : boolean, default True
        Aggregates data with mean or robust_mean
    stdcutoff : value, default 1
        Cutoff for outlier rejection, used in robust_mean
    """

    if b is None:
        b = 'b_field'
    if f is None:
        f = 'cantilever_frequency'
    if t is None:
        t = 'measurement_time'
    if temp is None:
        temp = 'thermometer_c'

    if type(data) is list:
        data = join_bscans(data)
    else:
        data = add_ramp_index(data)

    grouped = data.fillna(method='pad').groupby([b, 'ramp_index'])
    time = grouped[t].mean().unstack()
    files = grouped.file_index.mean().unstack()
    scatter = grouped[f].std().unstack()
    avg_temp = _np.mean(_np.mean(grouped[temp].mean()))

    if robust is True:
        raw = grouped[f].apply(_gp.robust_mean, stdcutoff=stdcutoff*scatter
                               .median().median(), **kwargs).unstack()
    else:
        raw = grouped[f].mean().unstack()

    return ReducedBScan(raw, scatter, time, files, avg_temp,
                        cantilever_object=cantilever_object)


class ReducedBScan:

    """
    Convenience class used to process, plot and store bscan data.

    'ReducedBScan' contains consistently named attributes for easy access to
    relevant properties of analyzed data. 'ReducedBScan' also contains
    convenience methods to smooth and plot data.

    A ReducedBScan object is instantiated with 'reduce_bscan'.

    Attributes
    ----------
    self.raw : pandas DataFrame
        Raw cantilever frequency row indexed by B field and column indexed
        by ramp index
    self.scatter : pandas DataFrame
        Standard deviation of frequency measurements at each field step. Row
        indexed by B field and column indexed by ramp index.
    self.t : pandas DataFrame
        The average of the timestamps at each field step. Row indexed by B
        field and column indexed by ramp index.
    self.temp : value
        Average temperature of entire measurement in Kelvin.
    self.ramps : numpy Array
        Array containing all unique ramp indices.
    self._files : pandas DataFrame
        File identifier. Used to untangle data joined using 'join_bscans'.

    Methods
    -------
    self.smooth_with
        Smooths self.raw with arbitrary function
    self.smooth_with_lowess
        Smooths self.raw with lowess
    self.drop_ramps
        Drops specified ramps
    self.plot_raw
        Plots raw data
    self.copy
        Instantiates another ReducedBScan object that is a copy of self
    """

    def __init__(self, f, scatter, t, files, temp, cantilever_object=None):
        self.raw = f
        self.scatter = scatter
        self.t = t
        self.temp = temp
        self._files = files
        self.ramps = self.raw.columns.values.astype('float')

        if cantilever_object is None:
            self.cantilever = _gp.Bunch()
            self.cantilever.w = 300e-6
            self.cantilever.l = 500e-6
            self.cantilever.t = 120e-9
            self.cantilever.f0 = f.mean().mean()
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
            self.cantilever = cantilever_object

    def _to_timeseries(self, attr):

        """
        Converts specified attribute to a timeseries.
        """

        timestamps = self.t.stack().values
        files = self._files.stack().values
        values = attr.stack().values
        ts = _pd.DataFrame(values, index=[timestamps, files]).unstack()\
            .sort_index().fillna(method='bfill')
        ts.index.names = ['measurement_time']
        ts.columns.names = ['file']

        return ts

    def _total_rings(self):
        pad_area = self.cantilever.w * \
            self.cantilever.ring.fraction * self.cantilever.l
        ring_area = self.cantilever.ring.spacing ** 2
        return pad_area / ring_area

    def _cantilever_k(self):
        au_mass = self.cantilever.ring.number * self.cantilever.ring.density \
            * self.cantilever.ring.thickness * self.cantilever.ring.dia * 4 \
            * self.cantilever.ring.linewidth

        cantilever_mass = self.cantilever.w * self.cantilever.l * \
            self.cantilever.t * self.cantilever.density
        cantilever_emass = (33 / 140) * cantilever_mass

        total_mass = cantilever_emass + au_mass
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
        dia0 = _np.sin(theta) * dia
        delta = w * _np.cos(theta) + t * _np.sin(theta)
        phi0 = 6.62607e-34 / 1.60218e-19
        return ((dia0 - delta) * (dia - w) / phi0,
                (dia0 + delta) * (dia + w) / phi0)

    def _lowess_window(self, abperiod=None, window=5, **kwargs):

        if abperiod is None:
            abperiod = _np.mean(self._ab_range(**kwargs)) ** -1

        b = self.raw.index.values
        span = max(b) - min(b)
        frac = (window * abperiod) / span

        if frac > 1:
            return 1
        else:
            return frac

    def smooth_with(self, method, **kwargs):

        """
        Applies an arbitrary smoothing function to self.raw

        The smoothing function is applied separately to each ramp. The
        result is treated as the "background" of the persistent current
        signal.

        Parameters
        ----------
        method : Any function in module namespace
            Must accept a pandas Series or numpy Array and return a numpy
            Array.
        **kwargs passed to 'method' function

        Returns an AggregatedBScan subclass.
        """

        fullbackground = self.raw[self.ramps].apply(method, **kwargs)
        linear = fullbackground.apply(_gp.polysmooth, order=1)
        background = fullbackground - linear
        f = self.raw[self.ramps] - linear
        df = self.raw[self.ramps] - fullbackground

        return AggregatedBScan(self, fullbackground, background, f, df)

    def smooth_with_lowess(self, frac=None, it=None, delta=None, **kwargs):

        """
        Smooths self.raw with local linear regression (lowess).

        The lowess smoothing is applied separately to each ramp. The
        result is treated as the "background" of the persistent current
        signal.

        When 'frac' is not passed as an argument, an estimate for the
        smoothing window size is generated from the default values for
        'abperiod' (Aharonov-Bohm period) and 'window' from the internal
        method _lowess_window. Optionally, we can pass **kwargs for specific
        ring parameters to better estimate the Aharonov-Bohm period.

        When 'frac' is passed as an argument, the window size calculated
        from 'abperiod' and 'window' is overridden.

        Returns an AggregatedBScan subclass.

        Parameters
        ----------
        frac: float
            Between 0 and 1. The fraction of the data used
            when estimating each y-value.
        it: int
            The number of residual-based reweightings to perform.
        delta: float
            Distance within which to use linear-interpolation
            instead of weighted regression.
        **kwargs passed to _lowess_window to calculate "frac" parameter
        """

        if frac is None:
            frac = self._lowess_window(**kwargs)

        fullbackground = self.raw[self.ramps]\
            .apply(_gp.lowess, frac=frac, it=it, delta=delta)
        linear = fullbackground.apply(_gp.polysmooth, order=1)
        background = fullbackground - linear
        f = self.raw[self.ramps] - linear
        df = self.raw[self.ramps] - fullbackground

        return AggregatedBScan(self, fullbackground, background, f, df)

    def drop_ramps(self, ramps_to_drop):

        """
        Deletes unwanted ramps. Self-explanatory, really.

        Accepts list or numpy Array, returns ReducedBScan object.
        """

        if type(ramps_to_drop) is int:
            ramps_to_drop = [ramps_to_drop]

        ramp_mask = _np.array([any(_np.array(ramps_to_drop) == ramp)
                              for ramp in self.ramps])
        self.ramps = self.ramps[~ramp_mask]

        self.scatter = self.scatter.filter(items=self.ramps)
        self.t = self.t.filter(items=self.ramps)
        self._files = self._files.filter(items=self.ramps)
        self.raw = self.raw.filter(items=self.ramps)
        return self

    def plot_scatter(self, timeseries=False, **kwargs):

        """
        Plots standard deviation of frequency data vs. b-field.
        """

        if timeseries is True:
            ax = self._to_timeseries(self.scatter).plot(**kwargs)
            ax.set_xlabel('Measurement Time [seconds]')
        else:
            ax = self.scatter.plot(**kwargs)
            ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Standard Deviation of Frequency data')
        ax.legend(self.ramps)

        _gp.rstyle(ax)

        _plt.show()
        return ax

    def plot_raw(self, timeseries=False, **kwargs):

        """
        Plots raw frequency data (background included) vs. b-field.
        """

        if timeseries is True:
            ax = self._to_timeseries(self.raw).plot(**kwargs)
            ax.set_xlabel('Measurement Time [seconds]')
        else:
            ax = self.raw.plot(**kwargs)
            ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Raw BScan data')
        ax.legend(self.ramps)

        _gp.rstyle(ax)

        _plt.show()
        return ax

    def copy(self):

        """
        Returns a copy of self as a new ReducedBScan object.
        """

        return ReducedBScan(self.raw, self.scatter, self.t)


class AggregatedBScan(ReducedBScan):

    """
    Convenience subclass of ReducedBScan.

    This subclass inhereits attributes and methods from ReducedBScan and
    adds new ones related to the existence of a guess for the background.
    Upon instantiation, AggregatedBScan aggregates the frequency shift
    data using mean. This can be changed using the self.aggregate_with
    method.

    An AggregatedBScan object is instantiated any time a smoothing
    method is run on a ReducedBScan object.

    Attributes
    ----------
    self.fullbackground : pandas DataFrame
        Output of smoothing function.
    self.background : pandas DataFrame
        Background with 1st order polynomial removed for visual clarity.
    self.f : pandas DataFrame
        self.raw with self.background removed to better visualize raw data.
    self.df : pandas DataFrame
        Frequency shift data. self.raw with self.fullbackground removed.
    self.ab : pandas Series
        Aggregated frequency shift data. Can be recalculated with
        self.aggregate_with method.

    Methods
    -------
    self.aggregate_with
        Runs arbitrary aggregation function on self.df
    self.drop_ramps
    self.plot_with_background
    self.plot_df
    self.plot_fft
    self.plot
    self.copy
    """

    def __init__(self, parent_bscan, fullbackground, background, f, df):
        self.raw = parent_bscan.raw
        self.t = parent_bscan.t
        self.temp = parent_bscan.temp
        self._files = parent_bscan._files
        self.scatter = parent_bscan.scatter
        self.ramps = parent_bscan.ramps
        self.fullbackground = fullbackground
        self.background = background
        self.f = f
        self.df = df
        self.cantilever = parent_bscan.cantilever

        self.sensitivity_factor()
        if type(self.df) is _pd.Series:
            self.di = self.convert_to_pc(self.df)
        elif type(self.df) is _pd.DataFrame:
            self.di = self.df.apply(self.convert_to_pc)
        self.ab = self.di.mean(axis=1)

    def aggregate_with(self, method, drop=None, **kwargs):

        """
        Aggregates bscan frequency shift data with the specified method.

        Can call any function in namespace that aggregates a numpy array
        and returns a single value.

        Returns an AggregatedBScan object.

        Parameters
        ----------
        method : Any function in module namespace
            Must accept a pandas Series or numpy Array and return a numpy
            Array.
        drop : index, default none
            ignores specified columns in aggregation
        **kwargs passed to 'method' function
        """

        if drop is None:
            self.ab = self.di.apply(method, axis=1, **kwargs)
        else:
            self.ab = self.di[_np.delete(self.ramps, drop)]\
                .apply(method, axis=1, **kwargs)

        return self

    def sensitivity_factor(self):
        angle = self.cantilever.angle
        nrings = self.cantilever.ring.number
        aring = self.cantilever.ring.dia ** 2
        f0 = self.cantilever.f0
        k = self.cantilever.k
        L = self.cantilever.l * (1 - self.cantilever.ring.fraction / 2)

        alpha = 1.377

        theta = angle * _np.pi / 180
        self.sensitivity = - _np.sqrt(nrings) * (f0 / (2 * k)) * \
            ((alpha / L) ** 2) * aring * (_np.cos(theta) ** 2 /
                                          _np.sin(theta))

    def convert_to_pc(self, df_series, sensitivity=None):

        if sensitivity is None:
            sensitivity = self.sensitivity

        b = df_series.index.values.astype(float)
        df = df_series.values.astype(float)
        return df / (sensitivity * b ** 2)

    def  _alpha_calculator(self, mode_number=None, pad_fraction=None):
        
        if mode_number is None:
            mode_number = self.cantilever.mode
        if pad_fraction is None:
            pad_fraction = self.cantilever.ring.fraction

        beta = _gp.data.cantilevers.mode_gamma(mode_number)
        a = -(_np.sin(beta) + _np.sinh(beta)) / (_np.cos(beta)+_np.cosh(beta))

        xx = _np.arange(0,1,0.001)
        alphaarr = beta * (_np.cos(beta * xx) - _np.cosh(beta * xx) - a * (_np.sin(beta * xx) + _np.sinh( beta * xx))) / (a * (_np.cos(beta) - _np.cosh(beta)) + _np.sin(beta) - _np.sinh(beta))
        alpha = _spi.integrate.cumtrapz(alphaarr[(_np.shape(xx)[0] * (1-pad_fraction)):])[-1] / (_np.shape(xx)[0] * pad_fraction)
        return alpha

    def drop_ramps(self, ramps_to_drop):

        """
        Deletes unwanted ramps. Self-explanatory, really.

        Accepts list or numpy Array, returns ReducedBScan object.
        """

        super().drop_ramps(ramps_to_drop)
        self.fullbackground = self.fullbackground.filter(items=self.ramps)
        self.background = self.background.filter(items=self.ramps)
        self.f = self.f.filter(items=self.ramps)
        self.df = self.df.filter(items=self.ramps)
        self.di = self.di.filter(items=self.ramps)
        return self

    def plot_with_background(self, timeseries=False, **kwargs):

        """
        Plots background and raw data with a 1st order polynomial removed.

        These plots help us better visualize the character of the raw
        frequency data and background subtraction. The drift is often
        so large compared to the signal that removing a first-order
        polynomial from both greatly improves clarity.
        """

        if timeseries is True:
            ax = self._to_timeseries(self.f).plot(**kwargs)
            self._to_timeseries(self.background).plot(ax=ax)
            ax.set_xlabel('Measurement Time [seconds]')
        else:
            ax = self.f.plot(**kwargs)
            self.background.plot(ax=ax)
            ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('BScan data--Linear background removed')
        ax.legend(self.ramps)

        _gp.rstyle(ax)

        _plt.show()
        return ax

    def plot_df(self, timeseries=False, **kwargs):

        """
        Plots frequency shift data.
        """

        if timeseries is True:
            ax = self._to_timeseries(self.df).plot(**kwargs)
            ax.set_xlabel('Measurement Time [seconds]')
        else:
            ax = self.df.plot(**kwargs)
            ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('BScan data--Background removed')
        ax.legend(self.ramps)

        _gp.rstyle(ax)

        _plt.show()
        return ax

    def plot_fft(self, dia=None, w=None, t=None, angle=None, **kwargs):

        """
        Plots the single-sided FFT of aggregrated frequency shift data.

        'plot_fft' utilizes scipy's fftpack helper functions 'fft' and
        'fftfreq' through grumpy.absfft. It plots the energy-normalized
        single-sided FFT with the magnetic-field frequency on the x-axis.

        Transparent vertical bars are overlaid on the FFT to indicate
        the 1st and 2nd Aharonov-Bohm frequency ranges calculated from the
        ring dimensions and orientation relative to the applied magnetic
        field. These spans are calculated by the internal
        ReducedBScan._ab_range method. If no ring parameters are passed,
        AB frequencies are calculated from the default values in
        the _ab_range method.

        Parameters
        ----------
        dia : value, default None
            Diameter of AB ring
        w : value, default None
            Linewidth of AB ring
        t : value, default None
            Thickness of AB ring
        angle : value, default None
            Angle of applied b-field relative to the plane of the ring
        **kwargs passed to pandas.Series.plot
        """

        self.ab = _pd.DataFrame(self.ab)
        fftdata = self.ab.apply(_gp.absfft)
        ax = fftdata.plot(**kwargs)
        ax.set_xlabel('Magnetic Field Frequency [1/Tesla]')
        ax.set_ylabel('<df> [Hz]')
        ax.set_title('FFT of BScan data')

        ab_range = self._ab_range(dia=dia, w=w, t=t, angle=angle)

        ax.axvspan(2 * ab_range[0], 2 * ab_range[1], alpha=.4,
                   color='yellow')
        ax.axvspan(ab_range[0], ab_range[1], alpha=.5, color='pink')

        _gp.rstyle(ax)

        _plt.show()
        return ax

    def plot(self, **kwargs):

        """
        Plots aggregated frequency shift data.
        """
        ax = self.ab.plot(**kwargs)
        ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Averaged BScan data')

        _gp.rstyle(ax)

        _plt.show()
        return ax

    def copy(self):

        """
        Returns a copy of self as a new AggregatedBScan object.
        """

        return AggregatedBScan(ReducedBScan(self.raw, self.scatter, self.t,
                               self._files), self.fullbackground,
                               self.background, self.f, self.df)
