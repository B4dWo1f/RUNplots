#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
This module contains all the useful functions for extracting information from
the wrfout files
"""

import os
here = os.path.dirname(os.path.realpath(__file__))
import log_help
import logging
LG = logging.getLogger(__name__)

import wrf
from netCDF4 import Dataset
import pathlib
import metpy.calc as mpcalc
#      # from metpy.units import units
import numpy as np
from . import util as ut
import datetime as dt
fmt = '%d/%m/%Y-%H:%M'

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))


def get_cache(ncfile):
   """
   Prepare WRF cache for faster post-processing
   """
   my_vars = ("P","PSFC","PB","PH","PHB","T","QVAPOR","HGT","U","V","W")
   LG.debug(f'Cached variables: {my_vars}')
   return wrf.extract_vars(ncfile, wrf.ALL_TIMES, (my_vars))


def getvar(ncfile,name,cache=None):
   """
   wrapper for wrf.getvar to include debug messages
   """
   aux = wrf.getvar(ncfile, name, cache=cache)
   try: LG.debug(f'{name}: [{aux.units}] {aux.shape}')
   except: LG.debug(f'{name}: {aux.shape}')
   return aux


@log_help.timer(LG)
def read_wrfout_info(fname):
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
   LG.info(f'WRFOUT file: {fname}')
   LG.info(f'WRFOUT folder: {wrfout_folder}')
   LG.info(f'Domain: {DOMAIN}')
 
   # Report here GFS batch and calculation time
   gfs_batch = open(f'{wrfout_folder}/batch.txt','r').read().strip()
   gfs_batch = dt.datetime.strptime(gfs_batch, fmt)
   LG.info(f'GFS batch: {gfs_batch}')

   # Get Creation date
   creation_date = pathlib.Path(fname).stat().st_mtime
   creation_date = dt.datetime.fromtimestamp(creation_date)
   LG.info(f'Data created: {creation_date.strftime(fmt)}')
 
   # Date in UTC
   # prefix to save files
   date = str(getvar(ncfile, 'times').values)
   date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
   LG.info(f'Forecast for: {date}')

   # Ref lat/lon
   reflat = ncfile.getncattr('CEN_LAT')
   reflon = ncfile.getncattr('CEN_LON')

   # bounds contain the bottom-left and upper-right corners of the domain
   # Notice that bounds will not be the left/right/top/bottom-most
   # latitudes/longitudes since the grid is only regular in Lambert Conformal
   bounds = wrf.geo_bounds(wrfin=ncfile)
   return ncfile, DOMAIN, bounds, reflat,reflon, wrfout_folder,\
          date, gfs_batch, creation_date


@log_help.timer(LG)
def clouds_rain(ncfile,prev=None,my_cache=None):
   """
   Cloud and rain related variables
   low_cloudfrac: [%] (ny,nx) percentage of low clouds cover
   mid_cloudfrac: [%] (ny,nx) percentage of mid clouds cover
   high_cloudfrac: [%] (ny,nx) percentage of high clouds cover
   rain: [mm] (ny,nx) mm of rain
   """
   # Relative Humidity____________________________________________[%] (nz,ny,nx)
   rh = getvar(ncfile, "rh", cache=my_cache)
   # Rain___________________________________________________________[mm] (ny,nx)
   rainc  = getvar(ncfile, "RAINC", cache=my_cache)
   rainnc = getvar(ncfile, "RAINNC", cache=my_cache)
   rain = rainc + rainnc
   if prev != None:
      rainc0  = getvar(prev, "RAINC", cache=my_cache)
      rainnc0 = getvar(prev, "RAINNC", cache=my_cache)
      rain0 = rainc0 + rainnc0
      rain -= rain0    # should be positive definite
      LG.info('Rain mm in 1 hour')
   else: LG.warning('Rain is cumulative')

   LG.debug(f'rain: {rain.shape}')
   # Clouds__________________________________________________________[%] (ny,nx)
   low_cloudfrac  = getvar(ncfile, "low_cloudfrac", cache=my_cache)
   mid_cloudfrac  = getvar(ncfile, "mid_cloudfrac", cache=my_cache)
   high_cloudfrac = getvar(ncfile, "high_cloudfrac", cache=my_cache)
   return low_cloudfrac, mid_cloudfrac, high_cloudfrac, rain


@log_help.timer(LG)
def wind(ncfile, cache=None):
   """
   Wind related variables
   u,v,w,u10,v10: [m/s] (nz,ny,nx) X,Y,Z components of the wind for every node
                                   in the model
   wspd,wdir: [m/s] (ny,nx) speed and direction of the wind at 10m agl
   wspd10,wdir10: [m/s] (ny,nx) speed and direction of the wind at 10m agl
   *Notice: w is not Earth-rotated
   """
   u,v = getvar(ncfile, 'uvmet', cache=cache)
   w = getvar(ncfile, 'wa', cache=cache)
   u10,v10 = getvar(ncfile, 'uvmet10', cache=cache)
   wspd,wdir = getvar(ncfile, 'uvmet_wspd_wdir', cache=cache)
   wspd10,wdir10 = getvar(ncfile, 'uvmet10_wspd_wdir', cache=cache)
   return u,v,w,u10,v10,wspd,wdir,wspd10,wdir10


@log_help.timer(LG)
def temperatures(ncfile,my_cache=None):
   """
   Temperature related variables
   tc: Model temperature in celsius
   tdc: Model dew temperature in celsius
   t2m: Temperature at 2m agl
   tmn: Soil temperature ???
   tsk: Skin temperature ???
   """
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = getvar(ncfile, "tc", cache=my_cache)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = getvar(ncfile, "td", cache=my_cache)
   # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
   t2m = getvar(ncfile, "T2", cache=my_cache)
   t2m = ut.convert_units(t2m, 'degC')
   LG.debug(f't2m: [{t2m.units}] {t2m.shape}')
   # Temperature Dew Point 2m above ground__________________________[°C] (ny,nx)
   td2m = getvar(ncfile, "td2")
   # SOIL TEMPERATURE AT LOWER BOUNDARY_________________________[K-->°C] (ny,nx)
   tmn = getvar(ncfile, "TMN", cache=my_cache)
   tmn = ut.convert_units(tmn, 'degC')
   LG.debug(f'tmn: [{tmn.units}] {tmn.shape}')
   # SKIN TEMPERATURE AT LOWER BOUNDARY_________________________[K-->°C] (ny,nx)
   tsk = getvar(ncfile, "TSK", cache=my_cache)
   tsk = ut.convert_units(tsk, 'degC')
   LG.debug(f'tsk: [{tsk.units}] {tsk.shape}')
   return tc,td,t2m,td2m,tmn,tsk



@log_help.timer(LG)
def model_variables(ncfile,my_cache=None):
   """
   Read model properties
   lats: (ny,nx) grid of latitudes
   lons: (ny,nx) grid of longitudes
   bounds: (ny,nx) bottom left and upper right corners of domain
   terrain: [m] (ny,nx) model topography
   heights: [m] (nz,ny,nx) height of each node of the model
   pressure: [hPa] (nz,ny,nx) pressure of each node of the model
   p: [hPa] (nz,ny,nx) perturbation pressure  of each node of the model
   pb: [hPa] (nz,ny,nx) base state pressure
   slp: [hPa] (ny,nx) sea level pressure
   bldepth: [m] (ny,nx) height of the BL
   """
   LG.info('Reading WRF data')
   # The domain is a rectangular (regular) grid in Lambert projection
   # Latitude, longitude___________________________________________[deg] (ny,nx)
   lats = getvar(ncfile, "lat", cache=my_cache)
   lons = getvar(ncfile, "lon", cache=my_cache)
   # bounds contain the bottom-left and upper-right corners of the domain
   # Notice that bounds will not be the left/right/top/bottom-most
   # latitudes/longitudes since the grid is only regular in Lambert
   bounds = wrf.geo_bounds(wrfin=ncfile)
   # Terrain topography used in the calculations_____________________[m] (ny,nx)
   terrain = getvar(ncfile, "ter", cache=my_cache) # = HGT
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   # Also called Geopotential Heights. heights[0,:,:] is the first level, 15m agl
   # XXX why 15 and not 10?
   heights = getvar(ncfile, "height", cache=my_cache) # = z
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = getvar(ncfile, "pressure", cache=my_cache)
   # Perturbation pressure_______________________________________[Pa] (nz,ny,nx)
   p = getvar(ncfile, "P", cache=my_cache)
   # Base state pressure_________________________________________[Pa] (nz,ny,nx)
   pb = getvar(ncfile, "PB", cache=my_cache)
   # Sea Level Pressure_____________________________________________[mb] (ny,nx)
   slp = getvar(ncfile, "slp", cache=my_cache)
   slp = ut.convert_units(slp,'mbar')
   # Planetary Boundary Layer Height_________________________________[m] (ny,nx)
   # Atmospheric Boundary layer thickness above ground
   bldepth = getvar(ncfile, "PBLH", cache=my_cache)
   # Atmospheric Boundary layer thickness above sea level
   # pblh = terrain + bldepth
   return lats,lons, bounds, terrain, heights, pressure,p,pb,slp,bldepth #,pblh


@log_help.timer(LG)
def all_properties(ncfile, prev=None, my_cache=None):
   """
   Read all the WRF properties and diagnostic variables.
   use_cache may accelerate around 20% performance
   """
   if my_cache == None: my_cache = get_cache(ncfile)
   else: LG.debug('Cache was provided')
   LG.debug('Reading WRF data')

   ### Read region data
   lats,lons, bounds, terrain, heights,\
             pressure,p,pb,slp,bldepth = model_variables(ncfile, my_cache)

   ## Temperature
   tc,td,t2m,td2m,tmn,tsk = temperatures(ncfile, my_cache)

   ## Wind
   u, v, w, u10, v10, wspd, wdir, wspd10, wdir10 = wind(ncfile, my_cache)

   ## Necessary for DrJack's routines
   # Surface sensible heat flux in________________________________[W/m²] (ny,nx)
   hfx = getvar(ncfile, "HFX", cache=my_cache) 
   # Cloud water mixing ratio________________________________[Kg/kg?] (nz,ny,nx)
   qcloud = getvar(ncfile, "QCLOUD", cache=my_cache)
   # Water vapor mixing ratio______________________________________[] (nz,ny,nx)
   qvapor = getvar(ncfile, "QVAPOR", cache=my_cache)

   low_frac, mid_frac, high_frac,\
rain = clouds_rain(ncfile,prev=prev,my_cache=my_cache)
   blcloudpct = low_frac + mid_frac + high_frac
   blcloudpct = np.clip(blcloudpct*100, None, 100)

   # LCL_____________________________________________________________[m] (ny,nx)
   LCL,_ = mpcalc.lcl(pressure[0,:,:], t2m, td2m)
   LCL = mpcalc.pressure_to_height_std(LCL) #np.array(lcl_p)*units.hPa)
   LCL = LCL.to('m')
   LCL = ut.maskPot0(LCL.magnitude, terrain,bldepth) #* LCL.units
   # CAPE_________________________________________________________[J/kg] (ny,nx)
   cape2d = getvar(ncfile, "cape_2d", cache=my_cache)
   MCAPE = cape2d[0,:,:]  # CAPE
   MCIN = cape2d[1,:,:]   # CIN
   LG.debug(f'CAPE: {MCAPE.shape}')
   LG.debug(f'CIN: {MCIN.shape}')
   LG.debug(f'LCL: {LCL.shape}')
   return lats,lons,u,v,w,u10,v10,wspd,wdir,wspd10,wdir10,pressure,heights,terrain,bldepth,hfx,qcloud,qvapor,tc,td,t2m,td2m,tsk,LCL,MCAPE,rain,low_frac,mid_frac,high_frac,blcloudpct


@log_help.timer(LG)
def sounding(ncfile,cache=None):
   """
   Extract the necessary variables for calculating a sounding
   """
   # Lats, Lons
   lats = getvar(ncfile, "lat", cache=cache)
   lons = getvar(ncfile, "lon", cache=cache)
   # Date in UTC
   # prefix to save files
   date = str(getvar(ncfile, 'times').values)
   date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = getvar(ncfile, "pressure", cache=cache)
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   # Also called Geopotential Heights. heights[0,:,:] is the first level, 15m agl   # XXX why 15 and not 10?
   heights = getvar(ncfile, "height", cache=cache)
   # Terrain topography used in the calculations_____________________[m] (ny,nx)
   terrain = getvar(ncfile, "ter", cache=cache)
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = getvar(ncfile, "tc", cache=cache)
   # print('tc',tc.shape)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = getvar(ncfile, "td", cache=cache)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td2m = getvar(ncfile, "td2")
   # print('td',td.shape)
   # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
   t2m = getvar(ncfile, "T2", cache=cache)
   t2m = ut.convert_units(t2m, 'degC')
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = getvar(ncfile, "ua", cache=cache)  # U wind component
   va = getvar(ncfile, "va", cache=cache)  # V wind component
   return date,lats,lons,terrain,pressure,heights,tc,td,t2m,td2m,ua,va


@log_help.timer(LG)
def meteogram(ncfile, cache=None):
   """
   return the properties necessary for plotting meteograms. These properties
   are the same as sounding plus wstar and hcrit
   date:
   lats:
   lons:
   terrain:
   pressure:
   heights:
   tc:
   td:
   t2m:
   td2m:
   ua, va:
   bldepth: height of the BL above ground level
   wstar:
   hcrit:
   """
   date,lats,lons,terrain,pressure,heights,\
                                         tc,td,t2m,td2m,ua,va = sounding(ncfile)
   bldepth = getvar(ncfile, "PBLH", cache=cache)
   hfx = getvar(ncfile, "HFX", cache=cache) 
   wstar = ut.calc_wstar( hfx, bldepth )
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth )
   return date,lats,lons,terrain,pressure,heights,tc,td,t2m,td2m,ua,va,bldepth,wstar,hcrit


def meteogram_hour(fname,lat,lon):
   ncfile = Dataset(fname)
   my_cache = get_cache(ncfile)
   # Lats, Lons
   lats = getvar(ncfile, "lat",cache=my_cache)
   lons = getvar(ncfile, "lon",cache=my_cache)
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = getvar(ncfile, "pressure", cache=my_cache)
   # Perturbation pressure_______________________________________[Pa] (nz,ny,nx)
   p = getvar(ncfile, "P", cache=my_cache)
   # Base state pressure_________________________________________[Pa] (nz,ny,nx)
   pb = getvar(ncfile, "PB", cache=my_cache)
   # Water vapor mixing ratio______________________________________[] (nz,ny,nx)
   qvapor = getvar(ncfile, "QVAPOR", cache=my_cache)
   # Planetary Boundary Layer Height_________________________________[m] (ny,nx)
   # Atmospheric Boundary layer thickness above ground
   bldepth = getvar(ncfile, "PBLH", cache=my_cache)
   # Surface sensible heat flux in________________________________[W/m²] (ny,nx)
   hfx = getvar(ncfile, "HFX", cache=my_cache) 
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   heights = getvar(ncfile, "height", cache=my_cache) # = z
   # Topography of the terrain ______________________________________[m] (ny,nx)
   terrain = getvar(ncfile, "ter", cache=my_cache) # = HGT
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = getvar(ncfile, "ua", cache=my_cache)  # U wind component
   va = getvar(ncfile, "va", cache=my_cache)  # V wind component
   wa = getvar(ncfile, "wa", cache=my_cache)  # W wind component
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = getvar(ncfile, "tc", cache=my_cache)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = getvar(ncfile, "td", cache=my_cache)
   # Thermal Updraft Velocity (W*)_________________________________[m/s] (ny,nx)
   wstar = ut.calc_wstar( hfx, bldepth )
   # Height of Critical Updraft Strength (hcrit)_____________________[m] (ny,nx)
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth)
   # Height of SFC.LCL_______________________________________________[m] (ny,nx)
   # Cu Cloudbase ~I~where Cu Potential > 0~P~
   zsfclcl = ut.calc_sfclclheight( pressure, tc, td, heights, terrain, bldepth )
   # OvercastDevelopment Cloudbase__________________________________[m?] (ny,nx)
   pmb = 0.01*(p.values+pb.values) # press is vertical coordinate in mb
   zblcl = ut.calc_blclheight(qvapor,heights,terrain,bldepth,pmb,tc)
   # Mask zsfclcl, zblcl________________________________________________________
   ## Mask Cu Pot > 0
   zsfclcldif = bldepth + terrain - zsfclcl
   null = 0. * zsfclcl
   # cu_base_pote = np.where(zsfclcldif>0, zsfclcl, null)
   zsfclcl = np.where(zsfclcldif>0, zsfclcl, null)
   ## Mask Overcast dev Pot > 0
   zblcldif = bldepth + terrain - zblcl
   null = 0. * zblcl
   # over_base_pote = np.where(zblcldif>0, zblcl, null)
   zblcl = np.where(zblcldif>0, zblcl, null)
   # Clouds__________________________________________________________[%] (ny,nx)
   low_cloudfrac  = getvar(ncfile, "low_cloudfrac", cache=my_cache)
   mid_cloudfrac  = getvar(ncfile, "mid_cloudfrac", cache=my_cache)
   high_cloudfrac = getvar(ncfile, "high_cloudfrac", cache=my_cache)
   # Thermalling Height______________________________________________[m] (ny,nx)
   hglider = np.minimum(np.minimum(hcrit,zsfclcl), zblcl)
   ## Point #####################################################################
   i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
   pblh  = bldepth[j,i].values
   hs = np.reshape(heights[:,j,i].values, (-1,1))
   u  = np.reshape(ua[:,j,i].values, (-1,1))
   v  = np.reshape(va[:,j,i].values, (-1,1))
   hcrit = hcrit[j,i]
   wstar = wstar[j,i]
   zblcl = zblcl[j,i]
   zsfclcl = zsfclcl[j,i]
   low_cloudfrac = low_cloudfrac[j,i].values
   mid_cloudfrac = mid_cloudfrac[j,i].values
   high_cloudfrac = high_cloudfrac[j,i].values
   gnd = terrain[j,i].values
   lat = lats[j,i].values
   lon = lons[j,i].values
   return lat,lon, hs, u, v, pblh, hcrit, wstar,gnd, zsfclcl, zblcl,\
          low_cloudfrac,mid_cloudfrac,high_cloudfrac


#def duplicate_first_row(M, value=None):
#   """
#   This function duplicates the first row like this:
#   1 1 1       1 1 1
#   2 2 2 ----> 2 2 2
#   3 3 3       3 3 3
#               3 3 3
#   """
#   first_row = M[0,:]
#   if value != None: first_row = first_row*0+value
#   return np.vstack([M,first_row])


#def get_meteogram(date0, lat0,lon0, data_fol, OUT_fol,place='', dom='d02',
#                                                                fout=None):
#   ## Read data
#   hours = list(range(8,22))
#   fmt_wrfout = '%Y-%m-%d_%H'
#   files = []
#   for h in hours:
#      if h == hours[-1]: h = hours[-2]   #XXX  workaround
#      date1 = date0.replace(hour=h) - UTCshift
#      files.append(f'{data_fol}/wrfout_{dom}_{date1.strftime(fmt_wrfout)}:00:00')
#   if not all([os.path.isfile(x) for x in files]):
#      print('Missing files!!!')
#      exit()
#   heights, windU,windV, BL,Hcrit,Wstar,Zcu,Zover = [],[],[],[],[],[],[],[]
#   PCT_low,PCT_mid,PCT_high = [],[],[]
#   for fname in files:
#      lat,lon, hs, u, v, pblh, hcrit, wstar, GND, cumulus, overcast,\
#      lowpct,midpct,highpct = get_data_hour(fname, lat0, lon0)
#      # hs, u, v, pblh, hcrit,wstar,GND,lat,lon = get_data_hour(fname, lat0, lon0)
#      heights.append(hs)
#      windU.append(u)
#      windV.append(v)
#      BL.append(pblh)
#      Hcrit.append(hcrit)
#      Wstar.append(wstar)
#      Zcu.append(cumulus)
#      Zover.append(overcast)
#      PCT_low.append(lowpct)
#      PCT_mid.append(midpct)
#      PCT_high.append(highpct)
#   U = np.hstack(windU) * 3.6
#   V = np.hstack(windV) * 3.6
#   heights = np.hstack(heights)
#   Wstar = np.hstack(Wstar)
#   BL = np.hstack(BL)
#   Zcu = np.hstack(Zcu)
#   Zover = np.hstack(Zover)
#   hours = np.hstack(hours)
#   # PCT_low = np.hstack(PCT_low)
#   # PCT_mid = np.hstack(PCT_mid)
#   # PCT_high = np.hstack(PCT_high)

#   ## Cut upper layers ##
#   Nup = 10
#   U = U[:-Nup,:]
#   V = V[:-Nup,:]
#   heights = heights[:-Nup,:]
#   ######################

#   ## Duplicate first row ##
#   U = duplicate_first_row(U)
#   V = duplicate_first_row(V)
#   heights = duplicate_first_row(heights,value=GND)#-10)
#   # Derived
#   S = np.sqrt(U*U + V*V)
#   X = np.array([hours for _ in range(U.shape[0])])
#   #########################

#   ## Plot
#   gs_plots = plt.GridSpec(3, 1, height_ratios=[2,17,1],hspace=0.,top=0.95,right=0.95,bottom=0.05)
#   gs_cbar  = plt.GridSpec(3, 1, height_ratios=[2,17,1],hspace=0.5,top=0.95,right=0.95, bottom=0)
#   fig = plt.figure()
#   # fig.subplots_adjust() #hspace=[0,0.2])
#   ax =  fig.add_subplot(gs_plots[1,:])   # meteogram
#   ax0 = fig.add_subplot(gs_plots[0,:], sharex=ax)  # clouds
#   ax1 = fig.add_subplot(gs_cbar[2,:])  # colorbar
#   ax.set_yscale('log')

#   ## % of low-mid-high-clouds
#   img_cloud_pct = np.vstack((PCT_low,PCT_mid,PCT_high))
#   Xcloud = np.array([hours for _ in range(img_cloud_pct.shape[0])])
#   Ycloud = 0*Xcloud.transpose() + np.array(range(img_cloud_pct.shape[0]))
#   Ycloud = Ycloud.transpose()
#   print(img_cloud_pct)
#   ax0.contourf(Xcloud,Ycloud,img_cloud_pct, origin='lower',
#                                             cmap='Greys', vmin=0, vmax=1)
#   # ax0.imshow(img_cloud_pct, origin='lower', cmap='Greys',aspect='auto')
#   # ax0.set_xticks([])
#   ax0.set_yticks(range(img_cloud_pct.shape[0]))
#   ax0.set_yticklabels(['low','mid','high'])
#   ax0.set_ylim(0,2)
#   plt.setp(ax0.get_xticklabels(), visible=False)
#   ax0.grid(False)
#   ## Plot Background Wind Speeds
#   C = ax.contourf(X, heights, S, levels=range(0,60,4),
#                                  vmin=0, vmax=60, cmap=WindSpeed,
#                                  extend='max',zorder=9)
#   ## alpha workaround
#   # rect = Rectangle((-1,-1),24,1e4,facecolor='white',zorder=10,alpha=0.5)
#   # ax.add_patch(rect)

#   ## Colorbar
#   cbar = fig.colorbar(C, cax=ax1, orientation="horizontal")

#   ## Plot BL and Thermals
#   thermal_color = np.array([255,127,0])/255 # (0.96862745,0.50980392,0.23921569)
#   BL_color      = np.array([255,205,142])/255 # (0.90196078,1., 0.50196078)
#   # thermal_color = (0.96862745,0.50980392,0.23921569)
#   # BL_color      = (0.90196078,1.,        0.50196078)
#   W = 0.6
#   ax.bar(hours,BL+GND,width=W, color=BL_color, ec=thermal_color,zorder=20)
#   ax.bar(hours,Hcrit-GND, width=W-0.15, bottom=GND,
#                           color=thermal_color,zorder=21)
#   ## Clouds
#   ax.bar(hours,Zover+100, bottom=Zover, width=1, color=(.4,.4,.4),
#                                                           zorder=21, alpha=0.8)
#   # ax.bar(hours,Zcu+100,bottom=Zcu,width=W+0.15, color=(.3,.3,.3), hatch='O', 
#   cu_top = np.where(Zcu>0,BL+GND-Zcu,-100)
#   ax.bar(hours,cu_top, bottom=Zcu, width=W+.2,color=(.3,.3,.3),hatch='O', 
#                                                           zorder=22, alpha=0.8)
#   # overcast_top = np.where(BL+GND > Zover,BL+GND-Zover,1000)
#   # ax.bar(hours,overcast_top, bottom=Zover, width=0.9, color=(.4,.4,.4),
#   #                                                       zorder=21, alpha=0.75)
#   # cu_top = np.where(BL+GND > Zcu,BL+GND-Zcu,Zcu+100)
#   # ax.bar(hours,cu_top, bottom=Zcu, width=W+0.15, color=(.3,.3,.3), hatch='O', 
#   #                                                       zorder=22, alpha=0.75)
#   ## Plot Wind barbs
#   ax.barbs(X,heights,U,V,length=6,zorder=30)
#   ## Plot Terrain Ground
#   terrain_color = (0.78235294, 0.37058824, 0.11568627)
#   rect = Rectangle((0,0), 24, GND, facecolor=terrain_color, zorder=29)
#   ax.add_patch(rect)
#   ax.text(0, 0, f'GND: {GND:.0f}', va='bottom', zorder=100,
#                                       transform=ax.transAxes)

#   ## Title
#   ax0.set_title(f'{lat:.3f},{lon:.3f}  -  {date0.date()}')

#   ## Axis setup
#   # X lim
#   ax.set_xlim(hours[1]-0.5, hours[-2]+0.5)
#   ax.set_xticks(hours[1:-1])
#   ax.set_xticklabels([f'{x}:00' for x in hours[1:-1]])
#   # Y lim
#   ymin = GND-75
#   ymax = np.max(BL)+GND+200
#   ax.set_ylim([ymin, ymax])
#   ax.yaxis.set_major_locator(MultipleLocator(500))
#   ax.yaxis.set_minor_locator(MultipleLocator(100))
#   ax.yaxis.set_major_formatter(ScalarFormatter())
#   ax.yaxis.set_minor_formatter(ScalarFormatter())
#   ax.set_ylabel('Height (m)')
#   # Grid
#   ax.grid(False)

#   # fig.tight_layout()
#   fig.savefig(fout,bbox_inches='tight')
#   return fout


#if __name__ == '__main__':
#   date_req = dt.datetime(2021,5,19)
#   lat,lon = 41.078854,-3.707029 # arcones ladera
#   lat,lon = 41.105178018195375, -3.712531733865551     # arcibes cantera
#   # lat,lon = 41.078887241417604, -3.7054138385286515  # arcones despegue
#   lat,lon = 41.131805855213194, -3.684117033819662 # pradena
#   lat,lon = 41.16434547255803, -3.571952688735032  # puerto cebollera
#   lat,lon = 41.172417,-3.617646 # somo
#   data_folder = '../../Documents/storage/WRFOUT/Spain6_1'
#   OUT_folder = '../../Documents/storage/PLOTS/Spain6_1'
#   place = ''
#   fout = 'meteogram.png'
#   dom = 'd02'
#   fname = get_meteogram(date_req, lat, lon, data_folder, OUT_folder, place, dom, fout)
#   print('Saved in:',fname)
