#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
This script will plot all the layers shown in the web http://raspuri.mooo.com/
Assumptions inherited from our way to run WRF (mainly file structure):
- wrfout files are outputed to wrfout_folder
- wrfout_folder should contain a batch.txt file containing the batch of GFS
data used for the wrfout files
"""

import log_help
import logging
LG = logging.getLogger(__name__)

# # WRF and maps
# from netCDF4 import Dataset
import wrf
# import metpy
# import rasterio
# from rasterio.merge import merge
# # My libraries
# from colormaps import WindSpeed, Convergencias, CAPE, Rain
# from colormaps import greys, reds, greens, blues
from . import util as ut
from . import extract
from metpy.units import units
import metpy.calc as mpcalc
# import plot_functions as PF   # My plotting functions
# # Standard libraries
from scipy.interpolate import interp1d
from configparser import ConfigParser, ExtendedInterpolation
# import pathlib
import numpy as np
import os
# import sys
here = os.path.dirname(os.path.realpath(__file__))
is_cron = bool( os.getenv('RUN_BY_CRON') )

import datetime as dt
fmt = '%d/%m/%Y-%H:%M'


@log_help.timer(LG)
@log_help.inout(LG)
def sounding(ncfile,lat,lon, pressure,tc,td,t2m,ua,va, terrain,lats,lons):
   """
   returns all the necessary properties for the provided coordinates (lat,lon)
   ncfile: ntcd4 Dataset from a wrfout file
   lat,lon: spatial coordinates for the sounding
   pressure: 3d model pressures
   tc: Model temperature in celsius
   tdc: Model dew temperature in celsius
   t2m: Model temperature 2m above ground
   ua: Model X wind (m/s)
   va: Model Y wind (m/s)
   terrain: model topography XXX obsolete
   lats: model latitudes grid
   lons: model longitudes grid
   """
   i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
   LG.info(f'({lat},{lon}) corresponds to ({j.values},{i.values}) node')
   # Get sounding data for specific location
   # Make unit aware
   lat = lats[j,i].values
   lon = lons[j,i].values
   LG.info(f'Closest point: ({lat},{lon})')
   p = pressure[:,j,i].metpy.quantify()
   tc = tc[:,j,i].metpy.quantify()
   tdc = td[:,j,i].metpy.quantify()
   t0 = t2m[j,i].metpy.quantify()
   u = ua[:,j,i] # .metpy.quantify()
   v = va[:,j,i] # .metpy.quantify()
   u = extract.convert_units(u, 'km/hour').metpy.quantify()
   v = extract.convert_units(v, 'km/hour').metpy.quantify()
   gnd = terrain[j,i]
   # Use metpy/pint quantity
   p = p.metpy.unit_array
   tc = tc.metpy.unit_array
   tdc = tdc.metpy.unit_array
   t0 = t0.metpy.unit_array
   u = u.metpy.unit_array
   v = v.metpy.unit_array
   gnd = gnd.metpy.unit_array
   # LCL
   lcl_p, lcl_t = mpcalc.lcl(p[0], t0, tdc[0])
   # Parcel Profile
   parcel_prof = mpcalc.parcel_profile(p, t0, tdc[0]) #.to('degC')
   ## Cumulus
   # base
   cu_base_p, cu_base_t = get_cloud_base(parcel_prof, p, tc, lcl_p, lcl_t)
   cu_base_m = mpcalc.pressure_to_height_std(cu_base_p)
   cu_base_m = cu_base_m.to('m')
   # top
   cu_top_p, cu_top_t = get_cloud_base(parcel_prof, p, tc)
   cu_top_m = mpcalc.pressure_to_height_std(cu_top_p)
   cu_top_m = cu_top_m.to('m')
   # Cumulus matrix
   ps, overcast, cumulus = get_cloud_extension(p,tc,tdc, cu_base_p,cu_top_p)
   rep = 3
   mats =  [overcast for _ in range(rep)]
   mats += [cumulus for _ in range(rep)]
   cloud = np.vstack(mats).transpose()
   Xcloud = np.vstack([range(2*rep) for _ in range(cloud.shape[0])])
   Ycloud = np.vstack([ps for _ in range(2*rep)]).transpose()
   return lat,lon,p,tc,tdc,t0,u,v,gnd,cu_base_p,cu_base_m,cu_base_t, Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof



@log_help.inout(LG)
def find_cross(left,right,p,tc,Ninterp=500):
   """
   finds the lowest (highest p) crossing point between left and right curves
   interp: interpolate the data for a more accurate crossing point
   """
   if Ninterp > 0:
      LG.debug('Interpolating with {Ninterp} points')
      ps = np.linspace(np.max(p),np.min(p),Ninterp)
      left = interp1d(p,left)(ps) * left.units
      right = interp1d(p,right)(ps) * right.units
      p = ps
   aux = left-right
   aux = (np.diff(aux/np.abs(aux)) != 0)*1   # manual implementation of sign
   ind, = np.where(aux==1)
   try: ind_cross = np.min(ind)
   except ValueError:
      LG.warning('Unable to find crossing point')
      ind_cross = 0
   p_base = p[ind_cross]
   t_base = right[ind_cross]
   return p_base, t_base


@log_help.inout(LG)
def get_cloud_base(parcel_profile,p,tc,lcl_p=None,lcl_t=None):
   """
   requires and keeps pint.units
   """
   LG.debug('Find cloud base')
   # Plot cloud base
   p_base, t_base = find_cross(parcel_profile, tc, p, tc)
   if type(lcl_p) != type(None) and type(lcl_t) != type(None):
      LG.debug(f'LCL is provided, using min(lcl,cu_base)')
      p_base = np.max([lcl_p.magnitude, p_base.magnitude]) * lcl_p.units
      t_base = np.max([lcl_t.magnitude, t_base.magnitude]) * lcl_t.units
   return p_base, t_base

@log_help.inout(LG)
def vertical_profile(N):
   """
   future modelling of the dependence of different kinds of clouds with
   the relative humidity at different levels
   """
   x0 = np.logspace(np.log10(.3),np.log10(7),N)
   t  = np.logspace(np.log10(.2),np.log10( 2),N)
   return x0, t

@log_help.inout(LG)
def get_cloud_extension(p,tc,td, cu_base,cu_top, threshold=.3, width=.2, N=500):
   """
   Calculate the extension of the clouds. Two kinds.
   - overcast: we'll consider clouds wherever tc-td < threshold (there's some
               smoothing controlled by width to account for our uncertainty).
               It returns two (N,) arrays with the pressures (altitudes) where
               there is (or isn't) cloud
   - cumulus: returns an array with shape p.shape with ones between cu_base and
              cu_top and zeroes elsewhere.
   returns 3 arrays with size (N,)
   ps: pressure levels for clouds
   overcast: proportional to non-convective cloud probability at every level
   cumulus: binary array with 1s where there are cumulus and 0s elsewhere
   """
   ## Clouds ############
   ps = np.linspace(np.max(p),np.min(p),N)
   tcs = interp1d(p,tc)(ps) * tc.units
   tds = interp1d(p,td)(ps) * td.units
   tdif = (tcs-tds).magnitude   # T-Td, proportional to Relative Humidity
   x0, t =  vertical_profile(N)
   overcast = ut.fermi(tdif, x0=x0,t=t)
   overcast = overcast/ut.fermi(0, x0=x0,t=t)
   cumulus = np.where((cu_base>ps) & (ps>cu_top),1,0)
   return ps, overcast, cumulus





def scalar_props(fname,section):
   """
   Return the data for plotting property. Intended to read from plots.ini
   """
   LG.info(f'Loading config file: {fname} for section {section}')
   # if not os.path.isfile(fname): return None
   config = ConfigParser(inline_comment_prefixes='#')
   config._interpolation = ExtendedInterpolation()
   config.read(fname)
   #XXX We shouldn't use eval
   factor = float(eval(config[section]['factor']))
   vmin   = float(eval(config[section]['vmin']))
   vmax   = float(eval(config[section]['vmax']))
   delta  = float(eval(config[section]['delta']))
   try: levels = config.getboolean(section, 'levels')
   except: levels = config[section].get('levels')
   if levels == False: levels = None
   elif levels != None:
      levels = levels.replace(']','').replace('[','')
      levels = list(map(float,levels.split(',')))
      levels = [float(l) for l in levels]
   else: levels = []
   cmap = config[section]['cmap']
   units = config[section]['units']
   return factor,vmin,vmax,delta,levels,cmap,units

def get_domain(fname):
   return fname.split('/')[-1].replace('wrfout_','').split('_')[0]
