"""
Functions for creating nice matplotlib plots
"""

import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt


def initPlot(fontsize=20, titlesize=20, labelsize=18, legendsize=14, usetex=True, fontfamily="sans-serif",
             topticks=True, rightticks=True, tickdirection="in"): 
    """
    Sets the basics of the plot

    Args:
        fontsize (int, optional): Fontsize of standard text in plot. Defaults to 20.
        titlesize (int, optional): Fontsize of title. Defaults to 20.
        labelsize (int, optional): Fontsize of tick-labels. Defaults to 18.
        legendsize (int, optional): Fontsize of legend. Defaults to 14.
        usetex (bool, optional): If true: Text is rendered as LaTeX. Defaults to True.
        fontfamily (str, optional): Font family to choose (serif, sans-serif, monospace). Defaults to "sans-serif".
        topticks (bool, optional): If true: Also displays xticks at the top of plots. Defaults to True.
        rightticks (bool, optional): If true: Also displays yticks at right of plots. Defaults to True.
        tickdirection (str, optional): Where ticks should point to (in or out). Defaults to "in".
    """
    properties =    {
        "text.usetex": usetex,
        "font.family": fontfamily,
        "axes.labelsize": labelsize,
        "font.size": fontsize,
        "legend.fontsize": legendsize,
        "xtick.labelsize": labelsize,
        "ytick.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "axes.facecolor": 'white',
        "xtick.top": topticks,
        "ytick.right": rightticks,
        "xtick.direction": tickdirection,
        "ytick.direction": tickdirection
    }

    plt.rcParams.update(properties)
    
def finalizePlot(ax, title="", outputFn="", showplot=True,  showlegend=True, tightlayout=True, legendcols=1, loc_legend="best", facecolor="white", dpi=300):
    """Finalizes Plots, saves it and shows it

    Args:
        ax (axis object from Matplotlib): Plot to be shown
        title (str, optional): TItle of plot. If empty: no title shown. Defaults to "".
        outputFn (str, optional): Filename to which plot should be saved. If empty: plot not saved. Defaults to "".
        showplot (bool, optional): If true: plot is shown. Defaults to True.
        showlegend (bool, optional): If true: legend is shown. Defaults to True.
        tightlayout (bool, optional): If true, matplotlibs option "tightlayout" is used. Defaults to True. Should be turned off for multi-panel plots.
        legendcols (int, optional): Number of columns in legend. Defaults to 1.
        loc_legend (str, optional): Location of legend. Defaults to "best".
        facecolor (str, optional): Background color of plot. Defaults to "white".
        dpi (int, optional): Number of dots per inch (i.e. resolution). Defaults to 300.
    """

    # Set title
    if(title != ""):
        ax.set_title(title)

    # Show legend
    if(showlegend):
        plt.legend(loc=loc_legend, ncol=legendcols)

    # Set tightlayout
    if(tightlayout):
        plt.tight_layout()

    # Save figure
    if(outputFn != ""):
        plt.savefig(outputFn, dpi=dpi, facecolor=facecolor)

    # Show plot
    if(showplot):
        plt.show()


def getColorList(numberColors, cmap="copper"):
    """Generates sequential list of colors from matplotlib colormaps

    Args:
        numberColors (int): number of colors to generate
        cmap (str, optional): name of colormap. Defaults to "copper". Check https://matplotlib.org/stable/users/explain/colors/colormaps.html for possible values.

    Returns:
        _type_: _description_
    """

    # Check if number of colors is integer
    if not isinstance(numberColors, int):
        raise ValueError("Number of colors needs to be an integer")
    
    # Create colormap
    cm=plt.get_cmap(cmap)
    
    # Create list of colors
    colors=[cm(i/numberColors) for i in range(numberColors)]
    return colors



# """
