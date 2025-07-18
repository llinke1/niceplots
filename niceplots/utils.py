"""
Functions for creating nice matplotlib plots
"""

import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from importlib_resources import files
import sys
import matplotlib as mpl
import shutil

def initPlot(version=2, colortype="categorical1", numbercolors=8):
    """
    Sets the basics of the plot by initializing the euclid stylesheet

    Args:
        version (int, optional): Which version of the stylesheet to use. Defaults to 2. Currently only 0, 1, and 2 are available.
        colortype (string, optional): How to set the default colors. Options are "categorical1", "categorical2", "categorical3", "sequential" and "diverging". All options are colorblind-friendly. Default is "categorical1", which initializes the Petroff color scheme.
        numbercolors (int, optional): How many colors to set as default colors when using "sequential" or "diverging". Is ignored, when a categorical color scheme is used. Default is 8.
    """

    if version > 2:
        raise ValueError("Currently only versions 0, 1, and 2 of style sheet are available")

    path=files('niceplots').joinpath(f"euclid_stylesheet_v{version}.mplstyle")
    plt.style.use(path)

        # Check if LaTeX is available
    latex_available = shutil.which("latex") is not None
    if not latex_available:
        print("⚠️ LaTeX not available. Rendering figures without it...")
        plt.rcParams["text.usetex"] = latex_available

    setDefaultColors(type=colortype, N=numbercolors)


    
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
        list: list of colors
    """

    # Check if number of colors is integer
    if not isinstance(numberColors, int):
        raise ValueError("Number of colors needs to be an integer")
    
    # Create colormap
    cm=plt.get_cmap(cmap)
    
    # Create list of colors
    colors=[cm(i/numberColors) for i in range(numberColors)]
    return colors

def setDefaultColors(type="categorical1", N=8):
    """ Set default colors. All schemes are colorblind friendly.
    The colors, depending on "type" are
    - type="sequential": Colors with increasing brightness taken from the matplotlib color map "copper"
    - type="diverging": Colors from blue to red, taken from the matplotlib color map "coolwarm"
    - type="categorical1": Colors from Petroff  (https://arxiv.org/pdf/2107.02270)
    - type="categorical2": Colors from Okabe & Ito (https://jfly.uni-koeln.de/color/#pallet)
    - type="categorical3": Colors from Tol (https://cran.r-project.org/web/packages/khroma/vignettes/tol.html#introduction)
    Args:
        type(str, optional): Type of color palette scheme. Can be "sequential", "diverging", or "categorical{i}" where i=1,2,3. Default is "categorical1"
        N (int, optional): How many colors to set as default colors when using "sequential" or "diverging". Is ignored, when a categorical color scheme is used. Default is 8.
    
    """
    if type=="sequential":
        colors=getColorList(N, cmap='copper')
    elif type=="diverging":
        colors=getColorList(N, cmap='coolwarm')
    elif type=="categorical1":
        # petroff palette based on https://arxiv.org/pdf/2107.02270
        colors= [(24/256, 69/256, 251/256), (255/256, 94/256, 2/256), (201/256, 31/256, 22/256), (200/256, 73/256, 169/256), (173/256, 173/256, 125/256), 
                      (134/256, 200/256, 221/256), (87/256, 141/256, 255/256), (101/256, 99/256, 100/256)]
    elif type=="categorical2":
        #okabe ito palette based on https://jfly.uni-koeln.de/color/#pallet
        colors= ['#000000', '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7']
    elif type=="categorical3":
        #tol vibrant palette based on https://cran.r-project.org/web/packages/khroma/vignettes/tol.html#introduction
        colors= ['#000000', '#ee7733', '#0077bb', '#33bbee', '#ee3377', '#cc3311', '#009988', '#bbbbbb']
    else:
        raise ValueError(f"{type} not implemented as color choice.")
    

    mpl.rcParams['axes.prop_cycle']=mpl.cycler(color=colors)
