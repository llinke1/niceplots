## niceplots

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

## License

Distributed under GNU GPL 3.0. See `LICENSE` for more information.

## Contact

Laila Linke - [laila.linke@uibk.ac.at](mailto: laila.linke@uibk.ac.at)

