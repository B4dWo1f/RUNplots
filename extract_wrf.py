#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import datetime as dt
from netCDF4 import Dataset
import wrf
import utils as ut
import drjack_interface as drj
# import mydrjack_num
import os
here = os.path.dirname(os.path.realpath(__file__))
HOME = os.getenv('HOME')
from pathlib import Path
fmt = '%d/%m/%Y-%H:%M'

def wrfout_info(fname):
   """
   Returns the file basic information
   fname: [str] path to file to be read
   Returns
   ncfile: [netCDF4.Dataset] WRF data
   DOMAIN: [str] corresponding domain name (d01, d02,...)
   bounds: [tuple] bottom left and upper right corners of the domain
   reflat,reflon: [float] Projection's reference latitude and longitude
                          more on Lambert Conformal:
               https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
   wrfout_folder: [str] path to folder containing all the wrfout files
   date: [datetime] valid date of data
   gfs_batch: [datetime] GFS batch used to calculate the data
   creation_date: [datetime] date when the calculations were made
                             (creation of the fname file)
   """
   # Read WRF data
   ncfile = Dataset(fname)

   # Get domain
   DOMAIN = ut.get_domain(fname)
   wrfout_folder = os.path.dirname(os.path.abspath(fname))
   # LG.info(f'WRFOUT file: {fname}')
   # LG.info(f'WRFOUT folder: {wrfout_folder}')
   # LG.info(f'Domain: {DOMAIN}')
 
   # Report here GFS batch and calculation time
   try:
       gfs_batch = open(f'{wrfout_folder}/batch.txt','r').read().strip()
       gfs_batch = dt.datetime.strptime(gfs_batch, fmt)
   except FileNotFoundError: gfs_batch = '???' 
   # LG.info(f'GFS batch: {gfs_batch}')

   # Get Creation date
   creation_date = Path(fname).stat().st_mtime
   creation_date = dt.datetime.fromtimestamp(creation_date)
   # LG.info(f'Data created: {creation_date.strftime(fmt)}')
 
   # Forecast date in UTC
   # prefix to save files
   date = str(wrf.getvar(ncfile, 'times').values)
   date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
   # LG.info(f'Forecast for: {date}')

   # Ref lat/lon
   reflat = ncfile.getncattr('CEN_LAT')
   reflon = ncfile.getncattr('CEN_LON')

   # bounds contain the bottom-left and upper-right corners of the domain
   # Notice that bounds will not be the left/right/top/bottom-most
   # latitudes/longitudes since the grid is only regular in Lambert Conformal
   bounds = wrf.geo_bounds(wrfin=ncfile)

   info = {'ncfile':ncfile,
           'domain': DOMAIN,
           'bounds': bounds,
           'reflat': reflat, 'reflon': reflon,
           'wrfout_folder':wrfout_folder,
           'date': date,
           'GFS_batch': gfs_batch,
           'creation_date': creation_date}

   return info

def get_rain(ncfile, cache={}):
   """
   Centralized function to extract rain form a provided ncfile
   """
   rainc  = wrf.getvar(ncfile, "RAINC",  cache=cache)
   rainnc = wrf.getvar(ncfile, "RAINNC", cache=cache)
   rainsh = wrf.getvar(ncfile, "RAINSH", cache=cache)
   return rainc + rainnc + rainsh

def wrf_vars(ncfile, prevnc=None, cache={}):
   """
   Extracts meteorological variables from a single WRF output file.

   Parameters:
     wrf_file_path (str): Path to the WRF output NetCDF file.

   Returns:
     dict: A dictionary containing:
      - 'umet', 'vmet': Earth-relative wind components [m/s], (nz, ny, nx)
      - 'w': Vertical wind component (Z) [m/s], (nz, ny, nx)
      - 'uvmet10': Earth-relative wind at 10m [m/s], (2, ny, nx)
      - 'wspd10','wdir10': Wind speed [m/s] and direction [°] at 10m, (ny, nx)
      - 'theta': Potential temperature [K], (nz, ny, nx)
      - 'tc': Temperature [°C], (nz, ny, nx)
      - 'td': Dewpoint temperature [°C], (nz, ny, nx)
      - 't2': 2m temperature [K], (ny, nx)
      - 'qvapor': Water vapor mixing ratio [kg/kg], (nz, ny, nx)
      - 'p': Pressure [hPa], (nz, ny, nx)
      - 'rh': Relative humidity [%], (nz, ny, nx) or None if unavailable
      - 'z': Model height AGL [m], (nz, ny, nx)
      - 'bldepth': PBL height [m], (ny, nx)
      - 'hfx': Surface sensible heat flux [W/m²], (ny, nx)
      - 'terrain': Terrain height [m], (ny, nx)
      - 'cape', 'cin', 'lcl', 'lfc': CAPE diagnostics [J/kg or m], (ny, nx)
   """
   # LG.info(f"Opening WRF file: {wrf_file_path}")

   # LG.info("Extracting basic atmospheric variables...")
   getvar = wrf.getvar
   heights = getvar(ncfile, "z", cache=cache)      # Model heights AGL (m)
   theta   = getvar(ncfile, "theta", cache=cache)  # Potential temperature
   t       = getvar(ncfile, "tc", cache=cache)     # Temperature (C)
   td      = getvar(ncfile, "td", cache=cache)     # Dewpoint (C)
   t2      = getvar(ncfile, "T2", cache=cache)     # 2m temperature (K)
   td2      = getvar(ncfile, "td2", cache=cache)     # 2m temperature (K)
   qvapor  = getvar(ncfile, "QVAPOR", cache=cache) # Water vapor mixing ratio (kg/kg)
   pmb     = getvar(ncfile, "pressure", cache=cache)  # Full pressure (hPa)

   # LG.info("Extracting wind variables...")
   uvmet = getvar(ncfile, "uvmet", cache=cache)   # (2, nz, ny, nx)
   w = getvar(ncfile, "wa", cache=cache)          # Vertical velocity (Z dir)
   # Surface wind
   uvmet10     = getvar(ncfile, "uvmet10", cache=cache)
   wspd_wdir10 = getvar(ncfile, "uvmet10_wspd_wdir", cache=cache)

   # LG.info("Extracting surface and boundary layer data...")
   bldepth = getvar(ncfile, "PBLH", cache=cache)  # PBL depth (m)
   hfx     = getvar(ncfile, "HFX", cache=cache)       # Sensible heat flux

   lats    = getvar(ncfile, "lat", cache=cache)
   lons    = getvar(ncfile, "lon", cache=cache)
   terrain = getvar(ncfile, "ter", cache=cache)

   # LG.info("Extracting Rain...")
   # rainc  = getvar(ncfile, "RAINC",  cache=cache)
   # rainnc = getvar(ncfile, "RAINNC", cache=cache)
   # rainsh = getvar(ncfile, "RAINSH", cache=cache)
   # rain = rainc + rainnc + rainsh
   rain = get_rain(ncfile, cache=cache)
   if not prevnc is None:
      rain0 = get_rain(prevnc)
      rain = rain - rain0
      # LG.info('Rain mm in 1 hour')
   else:
      # LG.warning('Rain is cumulative')
      print('Rain is cumulative')
   # LG.info("Extracting Cloud frac...")
   low_cloudfrac  = getvar(ncfile, "low_cloudfrac",  cache=cache)
   mid_cloudfrac  = getvar(ncfile, "mid_cloudfrac",  cache=cache)
   high_cloudfrac = getvar(ncfile, "high_cloudfrac", cache=cache)
   # LG.info("Extracting CAPE diagnostics...")
   cape, cin, lcl, lfc = getvar(ncfile, "cape_2d", cache=cache)

   try: rh = getvar(ncfile, "rh", cache=cache)
   except:
      # LG.warning("Relative humidity (RH) not available in file.")
      rh = None

   # LG.info("Extraction complete.")
   my_vars = {"uvmet": uvmet, "w": w,
              "uvmet10": uvmet10, "wspd_wdir10": wspd_wdir10,
              "theta": theta, "tc": t, "td": td, "t2m": t2, 'td2m':td2,
              "qvapor": qvapor, "rh": rh,
              "hfx": hfx,
              "p": pmb,
              "heights": heights,
              "bldepth": bldepth,
              "lats": lats, "lons": lons,
              "terrain": terrain,
              "rain": rain,
              "low_cloudfrac": low_cloudfrac,
              "mid_cloudfrac": mid_cloudfrac,
              "high_cloudfrac": high_cloudfrac,
              "cape": cape,
              "cin": cin,
              "lcl": lcl,
              "lfc": lfc}
   return my_vars

def drjack_vars(wrf_vars):
   """
   Computes derived quantities using Dr Jack's functions

   Parameters
   ----------
   u, v: [ndarray] (nz, ny, nx) 3D wind components (x and y) (m/s)
   w: [ndarray] (nz, ny, nx) 3D vertical velocity component (m/s)
   hfx: [ndarray] (ny, nx) 2D surface sensible heat flux (W/m²)
   pressure: [ndarray] (nz, ny, nx) 3D atmospheric full pressure field (hPa)
   heights: [ndarray] (nz, ny, nx) 3D model level heights (m)
   terrain: [ndarray] (ny, nx) 2D terrain height (m)
   bldepth: [ndarray] (ny, nx) 2D boundary layer depth (m)
   tc: [ndarray](nz, ny, nx) 3D temperature (°C)
   td: [ndarray] (nz, ny, nx) 3D dew point temperature (°C)
   qvapor: [ndarray] (nz, ny, nx) 3D water vapor mixing ratio (kg/kg)

   Returns
   -------
   info: [dict] Dictionary containing the following diagnostics:
      - 'wblmaxmin': Maximum up/down-draft in the BL (m/s)
      - 'wstar': Convective velocity scale (m/s)
      - 'hcrit': Critical climb height (m)
      - 'zsfclcl': Surface-based lifted condensation level height (m)
      - 'zblcl': BL-averaged lifted condensation level height (m)
      - 'hglider': Glider-usable thermal height estimate (m)
      - 'ublavgwind', 'vblavgwind': Boundary-layer-averaged wind components (m/s)
      - 'blwind': BL-averaged wind speed (m/s)
      - 'utop', 'vtop': Wind components at BL top (m/s)
      - 'bltopwind': Wind speed at BL top (m/s)

   Notes
   -----
   - Some transpositions are done internally to match Fortran-ordered routines.
   - The `hglider` parameter is computed as the maximum of hcrit and the minimum of
     zsfclcl and zblcl, following DrJack convention.
   """
   # Extracting WRF variables for usability
   u,v      = wrf_vars['uvmet']
   w        = wrf_vars['w']
   hfx      = wrf_vars['hfx']
   pressure = wrf_vars['p']
   heights  = wrf_vars['heights']
   terrain  = wrf_vars['terrain']
   bldepth  = wrf_vars['bldepth']
   tc       = wrf_vars['tc']
   td       = wrf_vars['td']
   qvapor   = wrf_vars['qvapor']

   wblmaxmin  = drj.calc_wblmaxmin(0, w, heights, terrain, bldepth)
   # print(ut.pretty_print_var(wblmaxmin))
   wstar      = drj.calc_wstar(hfx, bldepth)
   # print(ut.pretty_print_var(wstar))
   hcrit      = drj.calc_hcrit(wstar, terrain, bldepth, w_crit=1.143) # TODO w_crit=0
   # print(ut.pretty_print_var(hcrit))
   # print(hcrit.shape)
   zsfclcl    = drj.calc_sfclclheight(pressure, tc,td, heights,terrain, bldepth)
   # print(ut.pretty_print_var(zsfclcl))
   # print(zsfclcl.shape)
   zblcl      = drj.calc_blclheight(qvapor,heights,terrain,bldepth,pressure,tc)
   # print(ut.pretty_print_var(zblcl))
   # print(zblcl.shape)
   # exit()
   hglider = drj.calc_hglider(hcrit,zsfclcl,zblcl)
   # print(ut.pretty_print_var(hglider))

   ublavg= drj.calc_wind_blavg(u, heights, terrain, bldepth,
     name='ublavg', description='Boundary-layer-averaged wind U component')
   # print(ut.pretty_print_var(ublavg))
   vblavg= drj.calc_wind_blavg(v, heights, terrain, bldepth,
     name='vblavg', description='Boundary-layer-averaged wind V component')
   # print(ut.pretty_print_var(vblavg))
   blwind = drj.calc_Wspeed(ublavg, vblavg, name='blwind',
                              description='Boundary-layer-averaged wind speed')
   # blwind = np.sqrt( np.square(ublavgwind) + np.square(vblavgwind) )
   # print(ut.pretty_print_var(blwind))
   utop, vtop = drj.calc_bltopwind(u, v, heights, terrain, bldepth)
   # bltopwind = np.sqrt( np.square(utop) + np.square(vtop))
   bltopwind = drj.calc_Wspeed(utop, vtop, name='bltopwind',
                              description='Boundary-layer-averaged wind speed')
   info = {'wblmaxmin': wblmaxmin,
           'wstar': wstar,
           'hcrit': hcrit,
           'zsfclcl': zsfclcl,
           'zblcl': zblcl,
           'hglider': hglider,
           'ublavg': ublavg, 'vblavg': vblavg, 'blwind': blwind,
           'utop': utop, 'vtop': vtop, 'bltopwind': bltopwind}
   return info
