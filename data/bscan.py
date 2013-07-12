import numpy as _np
import grumpy as _gp
import pandas as _pd
import matplotlib.pyplot as _plt


def add_ramp_index(dataframe, offset=0, index_on='b_field'):

    """
    Indexes a dataframe column with a linearly ramped structure.

    "add_ramp_index" takes the second derivative of a specified dataframe
    column and looks for inflection points. Ramps are separated by these
    inflection points.

    Accepts a pandas DataFrame, returns a pandas DataFrame with an additional
    'ramp_index' column.

    Parameters
    ----------
    offset : integer, default 0
        Adds a constant offset to the ramp indices. Useful for joining
        datasets.
    index_on : string, default 'b_field'
        Indexes on specified column identifier.
    """

    dataframe['ramp_index'] = dataframe[index_on].fillna(method='ffill')
    dataframe['ramp_index'] = dataframe[index_on].diff().fillna(0)\
        .apply(_np.sign)
    nonzero = dataframe.ramp_index != 0
    dataframe.ramp_index[nonzero] = dataframe.ramp_index[nonzero]\
        .diff().fillna(0).apply(_np.abs).apply(_np.sign)
    dataframe.ramp_index = dataframe.ramp_index.cumsum() + offset
    return dataframe


def join_bscans(df_list, **kwargs):

    """
    Adds sequential ramp indices and concatenates bscans.

    Accepts a list of pandas DataFrames, returns a single pandas DataFrame
    with an additional 'ramp_index' column.

    Parameters
    ----------
    **kwargs passed to 'add_ramp_index'
        index_on : string, default 'b_field'
            Indexes on specified column identifier.
    """

    ramp_counter = 0
    for dataframe in df_list:
        dataframe = add_ramp_index(dataframe,
                                   offset=ramp_counter,
                                   **kwargs)
        ramp_counter += len(dataframe.ramp_index.unique())

    return _pd.concat(df_list)


def reduce_bscan(data, b=None, f=None, t=None, robust=True, cutoff=3):

    """
    Averages bscan data at each field point.

    'reduce_bscan' first adds a ramp_index column to the input DataFrame
    or list of DataFrames.

    Internally, 'reduce_bscan' calls the pandas DataFrame 'groupby' method
    and creates buckets of points indexed by ramp number and b_field. This
    groups together all points taken per "field step."

    The resulting buckets are aggregated using either _np.mean or
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
    robust : boolean, default True
        Aggregates data with _np.mean or robust_mean
    cutoff : value, default 3
        Cutoff for outlier rejection, used in robust_mean
    """

    if b is None:
        b = 'b_field'
    if f is None:
        f = 'cantilever_frequency'
    if t is None:
        t = 'measurement_time'

    if type(data) is list:
        data = join_bscans(data)
    else:
        data = add_ramp_index(data)

    grouped = data.groupby([b, 'ramp_index'])
    time = grouped[t].std().unstack()
    scatter = grouped[f].std().unstack()

    if robust is True:
        raw = grouped[f].apply(_gp.robust_mean, stdcutoff=cutoff*scatter
                               .median().median()).unstack()
    else:
        raw = grouped[f].mean().unstack()

    return ReducedBScan(raw, scatter, time)


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
    self.ramps : numpy Array
        Array containing all unique ramp indices.

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

    def __init__(self, f, scatter, t):
        self.raw = f
        self.scatter = scatter
        self.t = t
        self.ramps = self.raw.columns.values.astype('float')

    def _lowess_window(self, abperiod=.1, window=5):
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

    def smooth_with_lowess(self, frac=None, abperiod=.1, window=5):

        """
        Smooths self.raw with local linear regression (lowess).

        The lowess smoothing is applied separately to each ramp. The
        result is treated as the "background" of the persistent current
        signal.

        When 'frac' is not passed as an argument, an estimate for the
        smoothing window size is generated from the default values for
        'abperiod' (Aharonov-Bohm period) and 'window'. These parameters
        are adjustable.

        When 'frac' is passed as an argument, the window size calculated
        from 'abperiod' and 'window' is overridden.

        Returns an AggregatedBScan subclass.

        Parameters
        ----------
        frac : value between 0 and 1, default None
            Fraction of data to span with filter window, overrides
            the window calculated by window*abperiod/span
        abperiod : value, default .1
            Size of an Aharonov-Bohm period in Tesla
        window : value, default 5
            Smoothing window size as a multiple of Aharonov-Bohm periods
        """

        if frac is None:
            frac = self._lowess_window(abperiod=abperiod, window=window)

        fullbackground = self.raw[self.ramps].apply(_gp.lowess, frac=frac)
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

        self.ramps = _np.delete(self.ramps, ramps_to_drop)
        self.scatter = self.scatter[self.ramps]
        self.t = self.t[self.ramps]
        self.raw = self.raw[self.ramps]
        return self

    def plot_scatter(self):

        """
        Plots standard deviation of frequency data vs. b-field.
        """

        fig = _plt.figure()
        ax = fig.add_subplot(111)
        x_data = self.raw.index.values
        ax.plot(x_data, self.scatter.values)
        ax.set_title('Raw BScan data')
        ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(self.ramps, loc='center left', bbox_to_anchor=(1, 0.5))

        _plt.show()

    def plot_raw(self):

        """
        Plots raw frequency data (background included) vs. b-field.
        """

        fig = _plt.figure()
        ax = fig.add_subplot(111)
        x_data = self.raw.index.values
        ax.plot(x_data, self.raw.values)
        ax.set_title('Raw BScan data')
        ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(self.ramps, loc='center left', bbox_to_anchor=(1, 0.5))

        _plt.show()

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
    data using _np.mean. This can be changed using the self.aggregate_with
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
    self.plot
    self.copy
    """

    def __init__(self, parent_bscan, fullbackground, background, f, df):
        self.raw = parent_bscan.raw
        self.t = parent_bscan.t
        self.scatter = parent_bscan.scatter
        self.ramps = parent_bscan.ramps
        self.fullbackground = fullbackground
        self.background = background
        self.f = f
        self.df = df
        self.ab = self.df.mean(axis=1)

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
            self.ab = self.df.apply(method, axis=1, **kwargs)
        else:
            self.ab = self.df[_np.delete(self.ramps, drop)]\
                .apply(method, axis=1, **kwargs)

        return self

    def drop_ramps(self, ramps_to_drop):

        """
        Deletes unwanted ramps. Self-explanatory, really.

        Accepts list or numpy Array, returns ReducedBScan object.
        """

        self.ramps = _np.delete(self.ramps, ramps_to_drop)
        self.scatter = self.scatter[self.ramps]
        self.t = self.t[self.ramps]
        self.raw = self.raw[self.ramps]
        self.fullbackground = self.fullbackground[self.ramps]
        self.background = self.background[self.ramps]
        self.f = self.f[self.ramps]
        self.df = self.df[self.ramps]
        return self

    def plot_with_background(self):

        """
        Plots background and raw data with a 1st order polynomial removed.

        These plots help us better visualize the character of the raw
        frequency data and background subtraction. The drift is often
        so large compared to the signal that removing a first-order
        polynomial from both greatly improves clarity.
        """

        fig = _plt.figure()
        ax = fig.add_subplot(111)
        x_data = self.f.index.values
        ax.plot(x_data, self.f.values)
        ax.plot(x_data, self.background.values)
        ax.set_title('BScan data--linear drift removed')
        ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(self.ramps, loc='center left', bbox_to_anchor=(1, 0.5))

        _plt.show()

    def plot_df(self):

        """
        Plots frequency shift data.
        """

        fig = _plt.figure()
        ax = fig.add_subplot(111)
        x_data = self.df.index.values
        ax.plot(x_data, self.df.values)
        ax.set_title('BScan data--drift removed')
        ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(self.ramps, loc='center left', bbox_to_anchor=(1, 0.5))

        _plt.show()

    def plot(self):

        """
        Plots aggregated frequency shift data.
        """

        fig = _plt.figure()
        ax = fig.add_subplot(111)
        x_data = self.ab.index.values
        ax.plot(x_data, self.ab.values)
        ax.set_title('Averaged BScan data')
        ax.set_xlabel('Applied B-Field [Tesla]')
        ax.set_ylabel('Frequency [Hz]')

        _plt.show()

    def copy(self):

        """
        Returns a copy of self as a new AggregatedBScan object.
        """

        return AggregatedBScan(ReducedBScan(self.raw, self.scatter, self.t),
                               self.fullbackground, self.background, self.f,
                               self.df)
