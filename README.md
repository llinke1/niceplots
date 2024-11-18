## niceplots 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14182015.svg)](https://doi.org/10.5281/zenodo.14182015)

This code can help in producing nice plots! It contains routines that change the default settings of ```matplotlib``` to larger fontsizes, better ticks, and using proper TeX rendering. 

## Contributors
* Laila Linke
* Lukas Hergt

## Prerequisites

Using this code requires:
* **python3**
* **matplotlib**

The example notebooks also require
* **numpy**

## Installing

Simply go to the root directory of this repository and execute
```pip install ./```

## Usage

For usage, it should be as simple as using

``` 
import niceplot.utils as nicepl
nicepl.initPlot()
```

before executing your own ```matplotlib```-based plotting routines.
See the folder ```examples/``` for some example notebooks. The notebooks also contain (hopefully) useful information on colorpalettes, figure sizes and accessibility.

## License and Attribution

The code is distributed with a GNU GPL 3.0 license, which allows you to do pretty much all you want with it, provided any resulting codes are distributed with the same license. 
However, if you use the code in a publication, please cite it using [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14182015.svg)](https://doi.org/10.5281/zenodo.14182015)
or use the `bibTeX`:
```
@software{laila_linke_2024_14182015,
  author       = {Laila Linke},
  title        = {llinke1/niceplots: First release},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.14182015},
  url          = {https://doi.org/10.5281/zenodo.14182015}
}
```


## Contact

Laila Linke - [laila.linke@uibk.ac.at](mailto: laila.linke@uibk.ac.at)

