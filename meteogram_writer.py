#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import warnings
warnings.filterwarnings("ignore", message="Converting non-nanosecond precision datetime values to nanosecond precision")


import numpy as np
# import datetime as dt
# from netCDF4 import Dataset
import wrf
# import myutil as ut
# import mydrjack as drj
# # import mydrjack_num
import os
here = os.path.dirname(os.path.realpath(__file__))
HOME = os.getenv('HOME')
# from pathlib import Path
# fmt = '%d/%m/%Y-%H:%M'
import xarray as xr

def make_meteogram_timestep(calcdata,lat,lon):
   # 1D vars
   umet10, vmet10 = calcdata.wrf_vars["uvmet10"]  # [y, x]
   umet10_pt = extract_point_profile(umet10, lat, lon, calcdata)
   vmet10_pt = extract_point_profile(vmet10, lat, lon, calcdata)

   wspd10, wdir10 = calcdata.wrf_vars["wspd_wdir10"]  # [y, x]
   wspd10_pt = extract_point_profile(wspd10, lat, lon, calcdata)
   wdir10_pt = extract_point_profile(wdir10, lat, lon, calcdata)

   t0      = calcdata.wrf_vars['t2m']
   t0      = extract_point_profile(t0, lat, lon, calcdata)
   td0     = calcdata.wrf_vars['td2m']
   td0     = extract_point_profile(td0, lat, lon, calcdata)
   # Rain
   rain = calcdata.wrf_vars["rain"]
   rain_pt = extract_point_profile(rain, lat, lon, calcdata)
   # Clouds
   # low
   low_cloudfrac = calcdata.wrf_vars["low_cloudfrac"]
   low_cloudfrac_pt = extract_point_profile(low_cloudfrac, lat, lon, calcdata)
   # mid
   mid_cloudfrac = calcdata.wrf_vars["mid_cloudfrac"]
   mid_cloudfrac_pt = extract_point_profile(mid_cloudfrac, lat, lon, calcdata)
   # high
   high_cloudfrac = calcdata.wrf_vars["high_cloudfrac"]
   high_cloudfrac_pt = extract_point_profile(high_cloudfrac, lat, lon, calcdata)
   # Terrain
   terrain = extract_point_profile(calcdata.wrf_vars["terrain"], lat, lon, calcdata)
   # DrJack vars (scalars)
   hglider = extract_point_profile(calcdata.drjack_vars["hglider"], lat, lon, calcdata)
   zsfclcl = extract_point_profile(calcdata.drjack_vars["zsfclcl"], lat, lon, calcdata)
   zblcl = extract_point_profile(calcdata.drjack_vars["zblcl"], lat, lon, calcdata)
   wstar = extract_point_profile(calcdata.drjack_vars["wstar"], lat, lon, calcdata)

   # Vertical profile
   umet, vmet = calcdata.wrf_vars["uvmet"]
   umet_pt = extract_point_profile(umet, lat, lon, calcdata)
   vmet_pt = extract_point_profile(vmet, lat, lon, calcdata)
   tc = extract_point_profile(calcdata.wrf_vars["tc"], lat, lon, calcdata)
   tc = np.array(tc)  # Ensure 1D shape
   rh = extract_point_profile(calcdata.wrf_vars["rh"], lat, lon, calcdata)
   rh = np.array(rh)  # Ensure 1D shape
   p = extract_point_profile(calcdata.wrf_vars["p"], lat, lon, calcdata)
   p = np.array(p)  # Ensure 1D shape
   heights = extract_point_profile(calcdata.wrf_vars["heights"], lat, lon, calcdata)
   heights = np.array(heights)  # Ensure 1D shape
   timestamp = np.datetime64(calcdata.date).astype('datetime64[m]')

   wspd_pt = np.sqrt(umet_pt**2 + vmet_pt**2)

   # Time info
   timestamp = np.datetime64(calcdata.date,'m')  # Or from wrfout filename

   ds = xr.Dataset(
       data_vars={
           "terrain_height":     (["time"], [terrain]),
           "rain":               (["time"], [rain_pt]),
           "low_cloudfrac":      (["time"], [low_cloudfrac_pt]),
           "mid_cloudfrac":      (["time"], [mid_cloudfrac_pt]),
           "high_cloudfrac":     (["time"], [high_cloudfrac_pt]),
           "umet10":             (["time"], [umet10_pt]),
           "vmet10":             (["time"], [vmet10_pt]),
           "wspd10":             (["time"], [wspd10_pt]),
           "wdir10":             (["time"], [wdir10_pt]),
           "t0":                 (["time"], [t0]),
           "td0":                (["time"], [td0]),
           "wstar":              (["time"], [wstar]),
           "hglider":            (["time"], [hglider]),
           "zsfclcl":            (["time"], [zsfclcl]),
           "zblcl":              (["time"], [zblcl]),
           "p":        (["time", "level"], np.atleast_2d(p)),
           "tc":       (["time", "level"], np.atleast_2d(tc)),
           "rh":       (["time", "level"], np.atleast_2d(rh)),
           "heights":  (["time", "level"], np.atleast_2d(heights)),
           "umet":     (["time", "level"], np.atleast_2d(umet_pt)),
           "vmet":     (["time", "level"], np.atleast_2d(vmet_pt)),
           "wspd":     (["time", "level"], np.atleast_2d(wspd_pt)),
       },
       coords={
           "time": [timestamp],
       },
       attrs={
           "location_lat": lat,
           "location_lon": lon,
       }
   )

   return ds




def append_to_meteogram(ds_new, filepath):
   """Appends a new timestep to the meteogram netCDF file using safe context management"""

   # Ensure minute-level time precision for consistency
   ds_new["time"] = ds_new["time"].astype("datetime64[m]")

   if not os.path.exists(filepath):
      encoding = {"time": {"units": "minutes since 2025-04-29 00:00:00", "calendar": "standard"}}
      ds_new.to_netcdf(filepath, mode='w', encoding=encoding)
   else:
      with xr.open_dataset(filepath) as ds_existing:
         # Ensure consistent time precision
         ds_existing["time"] = ds_existing["time"].astype("datetime64[m]")

         # Ensure we compare with the same type
         times_new = np.array(ds_new["time"].values, dtype="datetime64[m]")

         # Drop overlapping times from existing dataset
         mask = ~np.isin(ds_existing["time"].values.astype("datetime64[m]"), times_new)
         ds_existing_filtered = ds_existing.isel(time=mask)

         # Concatenate and sort
         ds_combined = xr.concat([ds_existing_filtered, ds_new], dim="time", compat="identical", combine_attrs="override")
         ds_combined = ds_combined.sortby("time")

      encoding = {"time": {"units": "minutes since 2025-04-29 00:00:00", "calendar": "standard"}}
      ds_combined.to_netcdf(filepath, mode='w', encoding=encoding)
      ds_combined.close()



def extract_point_profile(var, lat, lon, calcdata):
   """Interpolates var at given lat/lon from wrf_vars (or drjack_vars)"""
   # Use wrf-python's ll_to_xy, or use nearest neighbor for simplicity

   ncfile = calcdata.ncfile
   j,i = wrf.ll_to_xy(ncfile, lat, lon)
   i = i.values
   j = j.values
   # 3D or 2D variable
   if var.ndim == 3:
      return var[:, i, j]
   elif var.ndim == 2:
      return var[i, j]
   else: raise
