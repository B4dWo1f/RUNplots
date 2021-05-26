# RUNplots

This repo inherits many things from [this other repo](https://github.com/B4dWo1f/RUN/). The idea is to split the calculation of the WRF and the post-processing of the data so we can centralize changes in the post-processing and include them in other repos (web, Telegram Bot...) as submodules.

I've structured the code in the following way:

```
RUN_post
  ├── wrf_calcs
  │   ├── post_process.py
  │   ├── extract.py
  │   ├── drjack.py
  │   └── util.py
  ├── terrain_tif
  │   └── gebco_08_rev_elev_*
  └── plots
      ├── geography.py
      ├── sounding.py
      ├── meteogram.py
      └── colormaps.py
```

`wrf_calcs` contains any and all calculations that are necessary to generate the maps or other info-graphics. The scripts in this folder depend strongly on the [WRF-python](https://wrf-python.readthedocs.io/en/latest/) and [Metpy](https://unidata.github.io/MetPy/latest/index.html)libraries as well as a private-for-now library developed by Dr. John W. Glendening ([DrJack](http://www.drjack.info/)). DrJack's codes are privative and only to be made public "`Upon the death of the original creator and copyright holder`", nonetheless he gave us permission to provide a compiled python library so people can used while keeping the source code.

`plots` all the functions to plot maps, soundings, etc... any function in the scripts in this folder should be fed already processed and in the correct units. These scripts are based in [matplotlib](https://matplotlib.org/) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/).
