"""
Functions for creating nice matplotlib plots
"""

import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from importlib_resources import files
import sys

def initPlot(version=1):
    """
    Sets the basics of the plot by initializing the euclid stylesheet

    Args:
        version (int, optional): Which version of the stylesheet to use. Defaults to 1. Currently only 0 and 1 are available.
    """

    if version > 1:
        raise ValueError("Currently only versions 0 and 1 of style sheet are available")

    path=files('niceplots').joinpath(f"euclid_stylesheet_v{version}.mplstyle")
    plt.style.use(path)


    
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
