# RUNplots

This repo inherits many things from [this other repo](https://github.com/B4dWo1f/RUN/). The idea is to split the calculation of the WRF and the post-processing of the data so we can centralize changes in the post-processing and include them in other repos (web, Telegram Bot...) as submodules.

I've structured the code in the following way:

```
RUN_post
  ├── run_postprocess.py
  │       Main code for post-processing wrfout files. The flow of the process:
  │        - read wrfout file into a CalcData class
  │        - plot background layers (terrains, rivers, cities, takeoffs...)
  │        - plot 2d scalar maps (wind speed, wstar, wblmaxmin, ...)
  │        - plot 2d vector maps (wind direction, streamlines and wind barbs)
  │        - plot soundings and meteograms
  │        - TO-DO: Save data for comparison to stations
  ├── calc_data.py
  │       Contains the definition of CalcData. It's purpose is to deal with
  │        the WRF interface, calculate DrJack variables and other derived
  │        quantities
  ├── extract_wrf.py
  │       Contains three main functions
  │        - wrfout_info: read metadata and general fields from wrfout
  │        - wrf_vars: read WRF "explicit" variables using mainly `wrf-python`
  │        - drjack_vars: calculate DrJack's variables particularly useful for
  │                       paragliding diagnostics. This function relies heavily
  │                       on drjack_interface.py
  ├── drjack_interface.py
  │       Python-Fortran wrappers for using DrJack's subroutines
  ├── drjack_num.cpython-310-x86_64-linux-gnu.so
  │       DrJack's functions compiled via f2py for python use
  ├── [drjack_num.f90]
  │       Not publicly provided by express petition of DrJack
  ├── derived_quantities.py
  │       Custom functions for point vertical profiles, cloud base/top...
  ├── meteogram_writer.py
  │       Helper functions to store meteogram data for the day without the need
  │       to load each wrfout each time a new data point appears
  ├── log_help.py
  │       Helper functions for help logging. For each wrfout file 2 log files
  │       will be created in logs:
  │         logs/
  │          ├── run_postprocess_<domain>_GFS<gfsbatch>.log
  │          └── run_postprocess_<domain>_GFS<gfsbatch>.perform
  ├── utils.py
  │       Helper functions: check directories, parse file names for domain/date,
  │       load config files, etc...
  ├── configs
  │   ├── soundings_d0*.csv
  │   ├── cities.csv
  │   ├── takeoffs.csv
  │   │      csv files for points of interest
  │   ├── plots.ini
  │   │      This file contains the colormap for every scalar property plotted
  │   │      Format:
  │   │         [sfcwind]
  │   │         factor = 3.6 # units conversion, for instance m/s to km/h
  │   │         delta  = 4
  │   │         vmin   = 0
  │   │         vmax   = 60
  │   │         cmap   = WindSpeed  # cmaps defined in plots/colormaps.py
  │   │         units  = Km/h       # To appear in colorbar legend
  │   │         title  = Viento Superficie  # Title of the plot. obsolete?
  │   └── zooms.ini
  │          This file contains the borders of relevant regions for static
  │          zoom in the maps
  │          Format:
  │             [guadarrama]
  │             parent = d02   # domain to use
  │             left = -4.384    # left longitude
  │             right = -3.264   # right longitude
  │             bottom = 40.534  # bottom latitude
  │             top = 41.43      # top latitude
  ├── plots
  │   ├── __init__.py
  │   ├── web.py
  │   │      Contains all the functions to prepare and order the data to plot
  │   │      the web-related maps
  │   │       - generate_background
  │   │       - generate_scalars
  │   │       - generate_vectors
  │   ├── geography.py
  │   │      Contains the actual cartopy and matplotlib details for plotting
  │   ├── fields.py
  │   │      Contains the actual matplotlib details for plotting
  │   ├── sounding.py
  │   │      The main function is skew_t_plot. It requires a CalcData object
  │   │      as input as well as the required lat/lon for the sounding
  │   ├── meteogram.py
  │   │      The main function is plot_meteogram. It requires a day-populated
  │   │      ncfile produced by ../meteogram_writer.py
  │   ├── colormaps.py
  │   ├── utils.py
  │   └── styles
  │       └── RASP.mplstyle
  └── terrain_tif
      ├── gebco_08_rev_elev_B1_grey_geo.tif
      ├── gebco_08_rev_elev_B2_grey_geo.tif
      ├── gebco_08_rev_elev_C1_grey_geo.tif
      ├── gebco_08_rev_elev_C2_grey_geo.tif
      ├── gebco_08_rev_elev_D1_grey_geo.tif
      └── gebco_08_rev_elev_D2_grey_geo.tif
```


## Continuous running
`wrfout_watcher.sh` is a continous loop that checks `$WRFOUT_DIR` for files with name `wrfout_d01*`.
When a new file is found it executes in parallel `$MAIN_SCRIPT` on the `wrfout_d01*` and `wrfout_d02*` files.
Once both files have being processed, they are moved to `$PROCESSED_DIR`

In order to launch this script at boot up we use a systemd service: `wrfout_watcher.service` which also sets up the log and error files for `wrfout_watcher.sh`

The services in ubuntu should be placed under `~/.config/systemd/user/` but symlinking is not recommended since `systemctl --user disable <name>.service` deletes all symlinks in the folder.
In order to circumvent this issue I've made yet another service which will be triggered whenever `wrfout_watcher.service` is modified.
The way it works is the following:
 - `sync_wrfout_watcher.path` is a hook that triggers `sync_wrfout_watcher.service` whenever `wrfout_watcher.service` is modified
 - `sync_wrfout_watcher.service` executes `sync_wrfout_watcher.sh`
 - `sync_wrfout_watcher.sh` copies the modified `wrfout_watcher.service` to `~/.config/systemd/user/` and reloads the daemon
