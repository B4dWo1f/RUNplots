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
LG.setLevel(logging.DEBUG)

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

def get_thermals(hfx,bldepth,terrain):
   wstar = ut.calc_wstar( hfx, bldepth )
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth)
   return wstar,hcrit

@log_help.timer(LG)
# @log_help.inout(LG)
def meteogram(ncfile,lat,lon, pressure,height,tc,td,t2m,td2m,ua,va,wstar,hcrit,bldepth, terrain,lats,lons):
   """
   returns all the necessary properties for the provided coordinates (lat,lon)
   as well as calculate derived properties
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
   hs = height[:,j,i].metpy.quantify()
   tc = tc[:,j,i].metpy.quantify()
   tdc = td[:,j,i].metpy.quantify()
   t0 = t2m[j,i].metpy.quantify()
   td0 = td2m[j,i].metpy.quantify()
   u = ua[:,j,i] # .metpy.quantify()
   v = va[:,j,i] # .metpy.quantify()
   u = ut.convert_units(u, 'km/hour').metpy.quantify()
   v = ut.convert_units(v, 'km/hour').metpy.quantify()
   gnd = terrain[j,i]
   bldepth = bldepth[j,i]
   wstar = wstar[j,i]
   hcrit = hcrit[j,i]
   # Use metpy/pint quantity
   p = p.metpy.unit_array
   hs = hs.metpy.unit_array
   tc = tc.metpy.unit_array
   tdc = tdc.metpy.unit_array
   t0 = t0.metpy.unit_array
   td0 = td0.metpy.unit_array
   u = u.metpy.unit_array
   v = v.metpy.unit_array
   gnd = gnd.metpy.unit_array
   bldepth = bldepth.metpy.unit_array
   return lat,lon,p,hs,tc,tdc,t0,td0,u,v,gnd,bldepth,wstar,hcrit

@log_help.timer(LG)
def sounding(ncfile,lat,lon, pressure,tc,td,t2m,td2m,ua,va, terrain,lats,lons):
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
   td0 = td2m[j,i].metpy.quantify()
   u = ua[:,j,i] # .metpy.quantify()
   v = va[:,j,i] # .metpy.quantify()
   u = ut.convert_units(u, 'km/hour').metpy.quantify()
   v = ut.convert_units(v, 'km/hour').metpy.quantify()
   gnd = terrain[j,i]
   # Use metpy/pint quantity
   p = p.metpy.unit_array
   tc = tc.metpy.unit_array
   tdc = tdc.metpy.unit_array
   t0 = t0.metpy.unit_array
   td0 = td0.metpy.unit_array
   u = u.metpy.unit_array
   v = v.metpy.unit_array
   gnd = gnd.metpy.unit_array
   # LCL
   lcl_p, lcl_t = mpcalc.lcl(p[0], t0, td0)
   # Parcel Profile
   parcel_prof = mpcalc.parcel_profile(p, t0, td0)
   try: parcel_prof = parcel_prof.convert_units('degC')
   except: parcel_prof = parcel_prof.to('degC')
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
   return lat,lon,p,tc,tdc,t0,td0,u,v,gnd,cu_base_p,cu_base_m,cu_base_t, ps,overcast,cumulus,lcl_p,lcl_t,parcel_prof



# @log_help.inout(LG)
@log_help.timer(LG)
def find_cross(profile,tc,p,Ninterp=500):
   """
   Inputs should be arrays WITHOUT UNITS
   finds the lowest (highest p) crossing point between profile and tc curves
   Ninterp: [int] Number of interpolation points for the arrays.
            If Ninterp = 0 no interpolation is used
   """
   if Ninterp > 0:
      LG.debug('Interpolating with {Ninterp} points')
      ps = np.linspace(np.max(p),np.min(p),Ninterp)
      profile = interp1d(p,profile)(ps) #* profile.units
      tc = interp1d(p,tc)(ps) #* tc.units
      p = ps
   aux = profile-tc
   aux = (np.diff(aux/np.abs(aux)) != 0)*1   # manual implementation of sign
   ind, = np.where(aux==1)
   try: ind_cross = np.min(ind)
   except ValueError:
      LG.warning('Unable to find crossing point')
      ind_cross = 0
   p_base = p[ind_cross]
   t_base = tc[ind_cross]
   return p_base, t_base


# @log_help.inout(LG)
@log_help.timer(LG)
def get_cloud_base(parcel_profile,p,tc,lcl_p=None,lcl_t=None):
   """
   requires and keeps pint.units
   """
   LG.debug('Find cloud base')
   # Plot cloud base
   p_base, t_base = find_cross(parcel_profile, tc, p)
   if type(lcl_p) != type(None) and type(lcl_t) != type(None):
      t_base = t_base * lcl_t.units
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
   x0 = np.logspace(np.log10(.2),np.log10(7),N)
   t  = np.logspace(np.log10(.15),np.log10( 2),N)
   return x0, t

# @log_help.inout(LG)
@log_help.timer(LG)
def get_cloud_extension1(p,tc,td, t0, td0, threshold=.3, width=.2, N=0):
   """
   Calculate the extension of the clouds. Two kinds.
     - overcast: we'll consider overcast clouds wherever tc-td < threshold
                 (there's some smoothing controlled by width to account for our
                 uncertainty).
     - cumulus: we'll consider cumulus clouds between cu_base and cu_top
   Returns 3 arrays with size (N,)
   ps: pressure levels for clouds
   overcast: proportional to non-convective cloud probability at every level
   cumulus: binary array with 1s where there are cumulus and 0s elsewhere
   """
   lcl_p, lcl_t = mpcalc.lcl(p[0], t0, td0)
   parcel_prof = mpcalc.parcel_profile(p, t0, td0).metpy.quantify()
   parcel_prof = ut.convert_units(parcel_prof, 'degC')
   ## Cumulus
   # base
   cu_base_p, cu_base_t = get_cloud_base(parcel_prof, p, tc, lcl_p, lcl_t)
   cu_base_m = mpcalc.pressure_to_height_std(cu_base_p)
   # cu_base_m = cu_base_m.to('m')
   # top
   cu_top_p, cu_top_t = get_cloud_base(parcel_prof, p, tc)
   cu_top_p = cu_top_p * cu_base_p.units
   cu_top_t = cu_top_t * cu_base_t.units
   cu_top_m = mpcalc.pressure_to_height_std(cu_top_p)
   # cu_top_m = cu_top_m.to('m')
   ## Clouds ############
   if N > 0:
      ps = np.linspace(np.max(p),np.min(p),N)
      tcs = interp1d(p,tc)(ps) * tc.units
      tds = interp1d(p,td)(ps) * td.units
   else:
      N = len(p)
      ps = p
      tcs = tc
      tds = td
   tdif = (tcs-tds).values   # T-Td, proportional to Relative Humidity
   x0, t =  vertical_profile(N)
   overcast = ut.fermi(tdif, x0=x0,t=t)
   overcast = overcast/ut.fermi(0, x0=x0,t=t)
   cumulus = np.where((cu_base_p>ps) & (ps>cu_top_p),1,0)
   return ps, overcast, cumulus

@log_help.inout(LG)
def get_cloud_extension(p,tc,td, cu_base,cu_top, threshold=.3, width=.2, N=0):
   """
   Calculate the extension of the clouds. Two kinds.
     - overcast: we'll consider overcast clouds wherever tc-td < threshold
                 (there's some smoothing controlled by width to account for our
                 uncertainty).
     - cumulus: we'll consider cumulus clouds between cu_base and cu_top
   Returns 3 arrays with size (N,)
   ps: pressure levels for clouds
   overcast: proportional to non-convective cloud probability at every level
   cumulus: binary array with 1s where there are cumulus and 0s elsewhere
   """
   ## Clouds ############
   if N > 0:
      ps = np.linspace(np.max(p),np.min(p),N)
      tcs = interp1d(p,tc)(ps) * tc.units
      tds = interp1d(p,td)(ps) * td.units
   else:
      N = len(p)
      ps = p
      tcs = tc
      tds = td
   tdif = (tcs-tds).magnitude   # T-Td, proportional to Relative Humidity
   x0, t =  vertical_profile(N)
   overcast = ut.fermi(tdif, x0=x0,t=t)
   overcast = overcast/ut.fermi(0, x0=x0,t=t)
   cumulus = np.where((cu_base>ps) & (ps>cu_top),1,0)
   return ps, overcast, cumulus



def spddir2uv(wspd,wdir):
   """
   Return U,V components from wind speed and direction
   """
   u = -wspd*np.sin(np.radians(wdir))
   v = -wspd*np.cos(np.radians(wdir))
   return u,v

def drjacks_vars(u,v,w, hfx, pressure,heights, terrain, bldepth,tc, td,qvapor):
   wblmaxmin = ut.calc_wblmaxmin(0, w, heights, terrain, bldepth)
   wstar = ut.calc_wstar(hfx, bldepth)
   hcrit = ut.calc_hcrit(wstar, terrain, bldepth)
   zsfclcl = ut.calc_sfclclheight(pressure, tc, td, heights, terrain, bldepth)
   zblcl = ut.calc_blclheight(qvapor,heights,terrain,bldepth,pressure,tc)
   hglider = np.minimum(np.minimum(hcrit,zsfclcl), zblcl)
   hglider = np.maximum(hglider,terrain)

   # Pot > 0
   zsfclcl = ut.maskPot0(zsfclcl, terrain,bldepth)
   zblcl = ut.maskPot0(zblcl, terrain,bldepth)

   ublavgwind = ut.calc_blavg(u, heights, terrain, bldepth)
   vblavgwind = ut.calc_blavg(v, heights, terrain, bldepth)
   blwind = np.sqrt( np.square(ublavgwind) + np.square(ublavgwind) )
   LG.debug(f'uBLavg: {ublavgwind.shape}')
   LG.debug(f'vBLavg: {vblavgwind.shape}')
   utop, vtop = ut.calc_bltopwind(u, v, heights, terrain, bldepth)
   bltopwind = np.sqrt( np.square(utop) + np.square(vtop))
   LG.debug(f'utop: {utop.shape}')
   LG.debug(f'vtop: {vtop.shape}')
   return wblmaxmin,wstar,hcrit,zsfclcl,zblcl,hglider,ublavgwind,vblavgwind,blwind,utop,vtop,bltopwind

def wblmaxmin(heights,pblh,w):
   """
   python's implementation of DrJacks wblmaxmin calculation.
   Returns the biggest value (positive or negative) of the W component within
   the BL.
   heights: (nz,ny,nx) matrix of model levels
   pblh: (ny,nx) height of the BL
   w: (nz,ny,nx) W component of the wind
   """
   dif = heights-pblh
   # dif < 0 inside the BL
   # dif > 0 outside the BL
   M = np.where(dif<0.,self.w,0.)
   wmax = np.max(M,axis=0)
   wmin = np.min(M,axis=0)
   return np.where(wmax>np.abs(wmin), wmax, wmin)



#XXX this should be in plots/utils.py
@log_help.timer(LG)
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
   title = config[section]['title']
   return factor,vmin,vmax,delta,levels,cmap,units,title

def get_domain(fname):
   return fname.split('/')[-1].replace('wrfout_','').split('_')[0]
