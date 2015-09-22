import numpy as _np
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import matplotlib.pylab as _pylab
import grumpy.plot.husl as _husl


def huslgen(hue, sat=80, light=50):
    return _husl.husl_to_hex(hue, sat, light)


def lighten(color, factor=1.2):
    if type(color) is str:
        base_color = _husl.hex_to_husl(color)
    elif type(color) is tuple:
        base_color = _husl.rgb_to_husl(*color)
    base_color[-1] = base_color[-1] * factor
    return _husl.husl_to_hex(*base_color)


def rstyle(ax):
    """
    Styles x,y axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been
    carried out (needs to know final tick spacing)
    """
    # Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.90')
    ax.set_axisbelow(True)

    # Set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator((_pylab.MultipleLocator((_plt.xticks()[0][1]
                                - _plt.xticks()[0][0]) / 2.0)))
    ax.yaxis.set_minor_locator((_pylab.MultipleLocator((_plt.yticks()[0][1]
                                - _plt.yticks()[0][0]) / 2.0)))

    # Remove axis border
    for child in ax.get_children():
        if isinstance(child, _mpl.spines.Spine):
            child.set_alpha(0)

    # Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    # Remove the minor tick lines
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    #Only show bottom left ticks, pointing out of axis
    _plt.rcParams['xtick.direction'] = 'out'
    _plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if ax.legend_ is not None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)


def rhist(ax, data, **keywords):
    """
    Creates a histogram with default style parameters to look like ggplot2
    Is equivalent to calling ax.hist and accepts the same keyword parameters.
    If style parameters are explicitly defined, they will not be overwritten
    """

    defaults = {'facecolor': '0.3',
                'edgecolor': '0.28',
                'linewidth': '1',
                'bins': 100}

    for k, v in defaults.items():
        if k not in keywords:
            keywords[k] = v

    return ax.hist(data, **keywords)


def waterfall(the_dataframe, offset=.3):

    last = []
    line_handles = []
    the_dataframe = the_dataframe.fillna(method='pad', limit=1)
    scale = _np.abs(the_dataframe.max().max() - the_dataframe.min().min())
    fig, ax = _plt.subplots()

    for index, cname in enumerate(the_dataframe.columns):
        new = the_dataframe[cname].add(index * offset * scale)
        base_line, = ax.plot(new.index.values.astype(float), new, alpha=.8)
        line_handles.append(base_line)

        if index > 0:
            plotrange = the_dataframe.index.values.astype(float)
            base_color = _mpl.colors.colorConverter.to_rgb(base_line.
                                                           get_color())
            ax.fill_between(plotrange, last[plotrange].values.astype(float),
                            new[plotrange].values.astype(float),
                            facecolor=lighten(base_color),
                            alpha=.2)
        elif index == 0:
            first_color = _mpl.colors.colorConverter.to_rgb(base_line.
                                                            get_color())

        last = new

    legend = ax.legend(line_handles[::-1], the_dataframe.columns.values[::-1],
                       bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.)
    for legobjects in legend.legendHandles:
        legobjects.set_linewidth(3)

    plotrange = the_dataframe.index.values.astype(float)
    base_color = _mpl.colors.colorConverter.to_rgb(base_line.
                                                   get_color())

    ylimits = ax.get_ylim()
    xlimits = (the_dataframe.index.values.min(),
               the_dataframe.index.values.max())

    ax.fill_between(plotrange, ylimits[0],
                    the_dataframe[the_dataframe.columns.values[0]][plotrange].
                    values.astype(float),
                    facecolor=lighten(first_color),
                    alpha=.2)

    _plt.ylim(ylimits)
    _plt.xlim(xlimits)

    return ax
