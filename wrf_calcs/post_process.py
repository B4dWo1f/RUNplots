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
# import wrf
# import metpy
# import rasterio
# from rasterio.merge import merge
# # My libraries
# from colormaps import WindSpeed, Convergencias, CAPE, Rain
# from colormaps import greys, reds, greens, blues
# from . import util as ut
# import plot_functions as PF   # My plotting functions
# # Standard libraries
from configparser import ConfigParser, ExtendedInterpolation
# import pathlib
import numpy as np
# import os
# import sys
# here = os.path.dirname(os.path.realpath(__file__))
# is_cron = bool( os.getenv('RUN_BY_CRON') )

import datetime as dt
fmt = '%d/%m/%Y-%H:%M'


def extract_wind(ncfile,my_cache=None):
   """
   Wind related variables
   ua,va,wa: [m/s] (nz,ny,nx) X,Y,Z components of the wind for every node
                              in the model
   wspd10,wdir10: [m/s] (ny,nx) speed and direction of the wind at 10m agl
   """
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = wrf.getvar(ncfile, "ua", cache=my_cache)  # U wind component
   LG.debug(f'ua: [{ua.units}] {ua.shape}')
   va = wrf.getvar(ncfile, "va", cache=my_cache)  # V wind component
   LG.debug(f'va: [{va.units}] {va.shape}')
   wa = wrf.getvar(ncfile, "wa", cache=my_cache)  # W wind component
   LG.debug(f'wa: [{wa.units}] {wa.shape}')
   # Wind at 10m___________________________________________________[m/s] (ny,nx)
   wspd10,wdir10 = wrf.g_uvmet.get_uvmet10_wspd_wdir(ncfile)
   LG.debug(f'wspd10: [{wspd10.units}] {wspd10.shape}')
   LG.debug(f'wdir10: [{wdir10.units}] {wdir10.shape}')
   return ua,va,wa, wspd10,wdir10

def get_sounding_vars(ncfile,my_cache=None):
   """
   Extract the variables required for soundings
   """
   # Latitude, longitude___________________________________________[deg] (ny,nx)
   lats = wrf.getvar(ncfile, "lat", cache=my_cache)
   lons = wrf.getvar(ncfile, "lon", cache=my_cache)
   LG.debug(f'lats: [{lats.units}] {lats.shape}')
   LG.debug(f'lons: [{lons.units}] {lons.shape}')
   # Terrain topography used in the calculations_____________________[m] (ny,nx)
   terrain = wrf.getvar(ncfile, "ter", units='m', cache=my_cache) # = HGT
   LG.debug(f'terrain: [{terrain.units}] {terrain.shape}')
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = wrf.getvar(ncfile, "pressure", cache=my_cache)
   LG.debug(f'pressure: [{pressure.units}] {pressure.shape}')
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = wrf.getvar(ncfile, "tc", cache=my_cache)
   LG.debug(f'tc: [{tc.units}] {tc.shape}')
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = wrf.getvar(ncfile, "td", units='degC', cache=my_cache)
   LG.debug(f'td: [{td.units}] {td.shape}')
   # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
   t2m = wrf.getvar(ncfile, "T2", cache=my_cache).metpy.convert_units('degC')
   t2m.attrs['units'] = 'degC'
   LG.debug(f't2m: [{t2m.units}] {t2m.shape}')
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = wrf.getvar(ncfile, "ua", cache=my_cache)  # U wind component
   LG.debug(f'ua: [{ua.units}] {ua.shape}')
   va = wrf.getvar(ncfile, "va", cache=my_cache)  # V wind component
   LG.debug(f'va: [{va.units}] {va.shape}')
   return lats,lons,pressure,tc,td,t2m,terrain,ua,va

def sounding(lat,lon,lats,lons,date,ncfile,pressure,tc,td,t0,ua,va,
                                                 title='',fout='sounding.png'):
   """
   lat,lon: spatial coordinates for the sounding
   date: UTC date-time for the sounding
   ncfile: ntcd4 Dataset from the WRF output
   tc: Model temperature in celsius
   tdc: Model dew temperature in celsius
   t0: Model temperature 2m above ground
   ua: Model X wind (m/s)
   va: Model Y wind (m/s)
   fout: save fig name
   """
   LG.info('Starting sounding')
   i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
   # Get sounding data for specific location
   # h = heights[:,i,j]
   latlon = f'({lats[j,i].values:.3f},{lons[j,i].values:.3f})'
   nk,nj,ni = pressure.shape
   p = pressure[:,j,i]
   tc = tc[:,j,i]
   tdc = td[:,j,i]
   u = ua[:,j,i]
   v = va[:,j,i]
   t0 = t0[j,i]
   LG.info('calling skewt plot')
   PL.skewt_plot(p,tc,tdc,t0,date,u,v,fout=fout,latlon=latlon,title=title)

from scipy.interpolate import interp1d
def find_cross(left,right,p,tc,interp=True):
   if interp:
      ps = np.linspace(np.max(p),np.min(p),500)
      left = interp1d(p,left)(ps)
      right = interp1d(p,right)(ps)
      aux = (np.diff(np.sign(left-right)) != 0)*1
      ind, = np.where(aux==1)
      ind_cross = np.min(ind)
      # ind_cross = np.argmin(np.abs(left-right))
      p_base = ps[ind_cross]
      t_base = right[ind_cross]
   else:
      aux = (np.diff(np.sign(left-right)) != 0)*1
      ind, = np.where(aux==1)
      ind_cross = np.min(ind)
      p_base = p[ind_cross]
      t_base = tc[ind_cross]
   return p_base, t_base


def get_cloud_base(parcel_profile,p,tc,lcl_p,lcl_t):
   # Plot cloud base
   p_base, t_base = find_cross(parcel_profile, tc, p, tc, interp=True)
   p_base = np.max([lcl_p, p_base])
   t_base = np.max([lcl_t, t_base])
   return p_base, t_base

def overcast(p,tc,td):
   ## Clouds ############
   ps = np.linspace(np.max(p),np.min(p),500)
   tcs = interp1d(p,tc)(ps)
   tds = interp1d(p,tdc)(ps)
   x0 = 0.3
   t = 0.2
   overcast = fermi(tcs-tds, x0=x0,t=t)
   overcast = overcast/fermi(0, x0=x0,t=t)
   cumulus = np.where((p_base>ps) & (ps>parcel_cross[0]),1,0)


def extract_clouds_rain(ncfile,my_cache=None):
   """
   Cloud and rain related variables
   low_cloudfrac: [%] (ny,nx) percentage of low clouds cover
   mid_cloudfrac: [%] (ny,nx) percentage of mid clouds cover
   high_cloudfrac: [%] (ny,nx) percentage of high clouds cover
   rain: [mm] (ny,nx) mm of rain
   """
   # Relative Humidity____________________________________________[%] (nz,ny,nx)
   rh = wrf.getvar(ncfile, "rh", cache=my_cache)
   # Rain___________________________________________________________[mm] (ny,nx)
   rain = wrf.getvar(ncfile, "RAINC", cache=my_cache) + wrf.getvar(ncfile, "RAINNC", cache=my_cache)
   LG.debug(f'rain: {rain.shape}')
   # Clouds__________________________________________________________[%] (ny,nx)
   low_cloudfrac  = wrf.getvar(ncfile, "low_cloudfrac", cache=my_cache)
   mid_cloudfrac  = wrf.getvar(ncfile, "mid_cloudfrac", cache=my_cache)
   high_cloudfrac = wrf.getvar(ncfile, "high_cloudfrac", cache=my_cache)
   LG.debug(f'low cloud: {low_cloudfrac.shape}')
   LG.debug(f'mid cloud: {mid_cloudfrac.shape}')
   LG.debug(f'high cloud: {high_cloudfrac.shape}')
   return low_cloudfrac, mid_cloudfrac, high_cloudfrac, rain

def extract_temperature(ncfile,my_cache=None):
   """
   Temperature related variables
   tc: Model temperature in celsius
   tdc: Model dew temperature in celsius
   t2m: Temperature at 2m agl
   tmn: Soil temperature ???
   tsk: Skin temperature ???
   """
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = wrf.getvar(ncfile, "tc", cache=my_cache)
   LG.debug(f'tc: [{tc.units}] {tc.shape}')
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = wrf.getvar(ncfile, "td", units='degC', cache=my_cache)
   LG.debug(f'td: [{td.units}] {td.shape}')
   # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
   t2m = wrf.getvar(ncfile, "T2", cache=my_cache).metpy.convert_units('degC')
   t2m.attrs['units'] = 'degC'
   LG.debug(f't2m: [{t2m.units}] {t2m.shape}')
   # SOIL TEMPERATURE AT LOWER BOUNDARY_________________________[K-->°C] (ny,nx)
   tmn = wrf.getvar(ncfile, "TMN", cache=my_cache).metpy.convert_units('degC')
   tmn.attrs['units'] = 'degC'
   LG.debug(f'tmn: [{tmn.units}] {tmn.shape}')
   # SKIN TEMPERATURE AT LOWER BOUNDARY_________________________[K-->°C] (ny,nx)
   tsk = wrf.getvar(ncfile, "TSK", cache=my_cache).metpy.convert_units('degC')
   tsk.attrs['units'] = 'degC'
   LG.debug(f'tsk: [{tsk.units}] {tsk.shape}')
   return tc,td,t2m,tmn,tsk


def extract_model_variables(ncfile,my_cache=None):
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
   lats = wrf.getvar(ncfile, "lat", cache=my_cache)
   lons = wrf.getvar(ncfile, "lon", cache=my_cache)
   LG.debug(f'lats: [{lats.units}] {lats.shape}')
   LG.debug(f'lons: [{lons.units}] {lons.shape}')
   # bounds contain the bottom-left and upper-right corners of the domain
   # Notice that bounds will not be the left/right/top/bottom-most
   # latitudes/longitudes since the grid is only regular in Lambert
   bounds = wrf.geo_bounds(wrfin=ncfile)
   # Terrain topography used in the calculations_____________________[m] (ny,nx)
   terrain = wrf.getvar(ncfile, "ter", units='m', cache=my_cache) # = HGT
   LG.debug(f'terrain: [{terrain.units}] {terrain.shape}')
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   # Also called Geopotential Heights. heights[0,:,:] is the first level, 15m agl
   # XXX why 15 and not 10?
   heights = wrf.getvar(ncfile, "height", units='m', cache=my_cache) # = z
   LG.debug(f'heights: [{heights.units}] {heights.shape}')
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = wrf.getvar(ncfile, "pressure", cache=my_cache)
   LG.debug(f'pressure: [{pressure.units}] {pressure.shape}')
   # Perturbation pressure_______________________________________[Pa] (nz,ny,nx)
   p = wrf.getvar(ncfile, "P", cache=my_cache)
   LG.debug(f'P: [{p.units}] {p.shape}')
   # Base state pressure_________________________________________[Pa] (nz,ny,nx)
   pb = wrf.getvar(ncfile, "PB", cache=my_cache)
   LG.debug(f'PB: [{pb.units}] {pb.shape}')
   # Sea Level Pressure_____________________________________________[mb] (ny,nx)
   slp = wrf.getvar(ncfile, "slp", units='mb', cache=my_cache)
   LG.debug(f'SeaLevelPressure: [{slp.units}] {slp.shape}')
   # Planetary Boundary Layer Height_________________________________[m] (ny,nx)
   # Atmospheric Boundary layer thickness above ground
   bldepth = wrf.getvar(ncfile, "PBLH", cache=my_cache)
   LG.debug(f'PBLH: [{bldepth.units}] {bldepth.shape}')
   return lats,lons, bounds, terrain, heights, pressure,p,pb,slp,bldepth


def extract_all_properties(ncfile,use_cache=True):
   """
   Read all the WRF properties and diagnostic variables.
   use_cache may accelerate around 20% performance
   """
   if use_cache:
      my_vars = ("P","PSFC","PB","PH","PHB","T","QVAPOR","HGT","U","V","W")
      my_cache = wrf.extract_vars(ncfile, wrf.ALL_TIMES, (my_vars))
   else: my_cache = None
   LG.debug('Reading WRF data')

   ### Read region data
   lats,lons, bounds, terrain, heights,\
             pressure,p,pb,slp,bldepth = extract_model_variables(ncfile,my_cache)

   tc,td,t2m,tmn,tsk = extract_temperature(ncfile,my_cache)
   tdif = tsk-tmn
   LG.debug(f'tsk: {tsk.shape}')

   ua,va,wa, wspd10,wdir10 = extract_wind(ncfile,my_cache)
   # ua10 = -wspd10 * np.sin(np.radians(wdir10))
   # va10 = -wspd10 * np.cos(np.radians(wdir10))

   ## Necessary for DrJack's routines
   # Surface sensible heat flux in________________________________[W/m²] (ny,nx)
   hfx = wrf.getvar(ncfile, "HFX", cache=my_cache) 
   LG.debug(f'HFX: {hfx.shape}')

   # Cloud water mixing ratio________________________________[Kg/kg?] (nz,ny,nx)
   qcloud = wrf.getvar(ncfile, "QCLOUD", cache=my_cache)
   LG.debug(f'QCLOUD: {qcloud.shape}')

   # Water vapor mixing ratio______________________________________[] (nz,ny,nx)
   qvapor = wrf.getvar(ncfile, "QVAPOR", cache=my_cache)
   LG.debug(f'QVAPOR: {qvapor.shape}')

   # import matplotlib.pyplot as plt
   # try: plt.style.use('mystyle')
   # except: pass
   # fig, ax = plt.subplots()
   # ax1 = ax.twiny()
   # ax.plot( tc[:,171,234], pressure[:,171,234], label='curva de estado')
   # ax.plot( td[:,171,234], pressure[:,171,234], label='curva de rocio')
   # ax1.plot(rh[:,171,234], pressure[:,171,234], 'k--',label='RH')
   # ax.set_xlabel('Pressure')
   # ax.set_ylabel('Temperature')
   # ax1.set_ylabel('RH (%)')
   # ax.scatter([20 for _ in pressure[:,171,234]],pressure[:,171,234],c=rh[:,171,234],cmap='Greys')
   # ax.legend()
   # ax.set_ylim(1000,150)
   # ax.set_xlim(-80,25)
   # fig.tight_layout()
   # fig, ax = plt.subplots()
   # ax.plot(rh[:,171,234])
   # ax1 = ax.twinx()
   # dif = tc[:,171,234]-td[:,171,234]
   # ax1.plot(np.max(dif)-dif,'C1',label='$T_c - T_d$')
   # ax1.legend()
   # fig.tight_layout()
   # plt.show()
   # exit()

   low_cloudfrac,mid_cloudfrac,high_cloudfrac,rain = extract_clouds_rain(ncfile,my_cache)
   blcloudpct = low_cloudfrac + mid_cloudfrac + high_cloudfrac
   blcloudpct = np.clip(blcloudpct*100, None, 100)

   # CAPE_________________________________________________________[J/kg] (ny,nx)
   cape2d = wrf.getvar(ncfile, "cape_2d", cache=my_cache)
   MCAPE = cape2d[0,:,:]  # CAPE
   MCIN = cape2d[1,:,:]   # CIN
   LCL = cape2d[2,:,:]    # Cloud base when forced lifting occurs
   LG.debug(f'CAPE: {MCAPE.shape}')
   LG.debug(f'CIN: {MCIN.shape}')
   LG.debug(f'LCL: {LCL.shape}')

   return bounds,lats,lons,wspd10,wdir10,ua,va,wa, heights, terrain, bldepth,hfx,qcloud,pressure,tc,td,t2m,p,pb,qvapor,MCAPE,rain,blcloudpct,tdif,low_cloudfrac,mid_cloudfrac,high_cloudfrac

def drjacks_calculations(ncfile,wa,heights,terrain,pressure,p,pb,bldepth,hfx,qvapor,qcloud,tc,td,my_cache=None):
   ## Derived Quantities by DrJack ##############################################
   # Using utils wrappers to hide the transpose of every variable
   # XXX Probalby Inefficient
   # BL Max. Up/Down Motion (BL Convergence)______________________[cm/s] (ny,nx)
   wblmaxmin = ut.calc_wblmaxmin(0, wa, heights, terrain, bldepth)
   LG.debug(f'wblmaxmin: {wblmaxmin.shape}')

   # Thermal Updraft Velocity (W*)_________________________________[m/s] (ny,nx)
   wstar = ut.calc_wstar( hfx, bldepth )
   LG.debug(f'W*: {wstar.shape}')

   # BLcwbase________________________________________________________[m] (ny,nx)
   # laglcwbase = 0 --> height above sea level
   # laglcwbase = 1 --> height above ground level
   laglcwbase = 0
   # criteriondegc = 1.0
   maxcwbasem = 5486.40
   cwbasecriteria = 0.000010
   blcwbase = ut.calc_blcloudbase( qcloud,  heights, terrain, bldepth,
                                   cwbasecriteria, maxcwbasem, laglcwbase)
   LG.debug(f'blcwbase: {blcwbase.shape}')

   # Height of Critical Updraft Strength (hcrit)_____________________[m] (ny,nx)
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth)
   LG.debug(f'hcrit: {hcrit.shape}')

   # Height of SFC.LCL_______________________________________________[m] (ny,nx)
   # Cu Cloudbase ~I~where Cu Potential > 0~P~
   zsfclcl = ut.calc_sfclclheight( pressure, tc, td, heights, terrain, bldepth )
   LG.debug(f'zsfclcl: {zsfclcl.shape}')

   # OvercastDevelopment Cloudbase_______________________________[m?] (nz,ny,nx)
   pmb = 0.01*(p.values+pb.values) # press is vertical coordinate in mb
   zblcl = ut.calc_blclheight(qvapor,heights,terrain,bldepth,pmb,tc)
   LG.debug(f'zblcl: {zblcl.shape}')

   # Thermalling Height______________________________________________[m] (ny,nx)
   hglider = np.minimum(np.minimum(hcrit,zsfclcl), zblcl)
   LG.debug(f'hglider: {hglider.shape}')

   # Mask zsfclcl, zblcl________________________________________________________
   ## Mask Cu Pot > 0
   zsfclcldif = bldepth + terrain - zsfclcl
   null = 0. * zsfclcl
   # cu_base_pote = np.where(zsfclcldif>0, zsfclcl, null)
   zsfclcl = np.where(zsfclcldif>0, zsfclcl, null)
   LG.debug(f'zsfclcl mask: {zsfclcl.shape}')

   ## Mask Overcast dev Pot > 0
   zblcldif = bldepth + terrain - zblcl
   null = 0. * zblcl
   # over_base_pote = np.where(zblcldif>0, zblcl, null)
   zblcl = np.where(zblcldif>0, zblcl, null)
   LG.debug(f'zblcl mask: {zblcl.shape}')

   # BL Avg Wind__________________________________________________[m/s?] (ny,nx)
   # uv NOT rotated to grid in m/s
   uv = wrf.getvar(ncfile, "uvmet", cache=my_cache)
   uEW = uv[0,:,:,:]
   vNS = uv[1,:,:,:]
   ublavgwind = ut.calc_blavg(uEW, heights, terrain, bldepth)
   vblavgwind = ut.calc_blavg(vNS, heights, terrain, bldepth)
   LG.debug(f'uBLavg: {ublavgwind.shape}')
   LG.debug(f'vBLavg: {vblavgwind.shape}')
   
   # BL Top Wind__________________________________________________[m/s?] (ny,nx)
   utop,vtop = ut.calc_bltopwind(uEW, vNS, heights,terrain,bldepth)
   LG.debug(f'utop: {utop.shape}')
   LG.debug(f'vtop: {vtop.shape}')
   return wblmaxmin, wstar, blcwbase, hcrit, zsfclcl, zblcl, hglider, ublavgwind, vblavgwind, utop,vtop


def get_info(ncfile):
   """
   XXX crappy workaround to explore the wrfout files
   """
   # Calculation parameters
   for x in ncfile.ncattrs():
      print(x)
      print(ncfile.getncattr(x))
      print('')
   # all WRF variables
   for v,k in ncfile.variables.items():
      print(v)
      # print(k.ncattrs())
      try: print(k.getncattr('description'))
      except: print('Description:')
      try: print(k.getncattr('units'))
      except: print('units: None')
      # print(k.dimensions)
      print(k.shape)
      print('')
   print('*******')


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

# def post_process_file(INfname,OUT_folder='plots'):
#    # Get domain
#    DOMAIN = get_domain(INfname)
#    wrfout_folder = os.path.dirname(os.path.abspath(INfname))
#    LG.info(f'WRFOUT file: {INfname}')
#    LG.info(f'WRFOUT folder: {wrfout_folder}')
#    LG.info(f'Domain: {DOMAIN}')
# 
#    # Report here GFS batch and calculation time
#    gfs_batch = open(f'{wrfout_folder}/batch.txt','r').read().strip()
#    gfs_batch = dt.datetime.strptime(gfs_batch, fmt)
#    LG.info(f'GFS batch: {gfs_batch}')
# 
#    # Get UTCshift
#    UTCshift = dt.datetime.now() - dt.datetime.utcnow()
#    UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))
#    LG.info(f'UTC shift: {UTCshift}')
# 
#    # Get Creation date
#    creation_date = pathlib.Path(INfname).stat().st_mtime
#    creation_date = dt.datetime.fromtimestamp(creation_date)
#    LG.info(f'Data created: {creation_date.strftime(fmt)}')
# 
#    # Read WRF data
#    ncfile = Dataset(INfname)
# 
#    # Date in UTC
#    # prefix to save files
#    date = str(wrf.getvar(ncfile, 'times').values)
#    date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
#    LG.info(f'Forecast for: {date}')
#    date_label = 'valid: ' + date.strftime( fmt ) + 'z\n'
#    date_label +=  'GFS: ' + gfs_batch.strftime( fmt ) + '\n'
#    date_label += 'plot: ' + creation_date.strftime( fmt+' ' )
#    
#    # Variables for saving outputs
#    OUT_folder = '/'.join([OUT_folder,DOMAIN,date.strftime('%Y/%m/%d')])
#    com = f'mkdir -p {OUT_folder}'
#    LG.warning(com)
#    os.system(com)
#    HH = date.strftime('%H%M')
#    
#    reflat = ncfile.getncattr('CEN_LAT')
#    reflon = ncfile.getncattr('CEN_LON')
#    
#    ## READ ALL VARIABLES
#    bounds, lats,lons,wspd10,wdir10,ua,va,wa, heights, terrain, bldepth,\
#    hfx,qcloud,pressure,tc,td,t2m,p,pb,qvapor,MCAPE,rain,blcloudpct,tdif,\
#    low_cloudfrac,mid_cloudfrac,high_cloudfrac = extract_all_properties(ncfile)
# 
#    # useful to setup the extent of the maps
#    left   = bounds.bottom_left.lon
#    right  = bounds.top_right.lon
#    bottom = bounds.bottom_left.lat
#    top    = bounds.top_right.lat
#    # left   = np.min(wrf.to_np(lons))
#    # right  = np.max(wrf.to_np(lons))
#    # bottom = np.min(wrf.to_np(lats))
#    # top    = np.max(wrf.to_np(lats))
# 
#    ## Derived Quantities
#    ua10 = -wspd10 * np.sin(np.radians(wdir10))
#    va10 = -wspd10 * np.cos(np.radians(wdir10))
#    
#    wblmaxmin, wstar, blcwbase, hcrit, zsfclcl, zblcl, hglider, ublavgwind, vblavgwind, utop,vtop = drjacks_calculations(ncfile,wa,heights,terrain,pressure,p,pb,bldepth,hfx,qvapor,qcloud,tc,td)
#    blwind = np.sqrt(ublavgwind*ublavgwind + vblavgwind*vblavgwind)
#    bltopwind = np.sqrt(utop*utop + vtop*vtop)
#    LG.debug(f'BLwind: {blwind.shape}')
#    LG.debug(f'BLtopwind: {bltopwind.shape}')
#    LG.info('WRF data read')
# 
#    ##############################################################################
#    #                                    Plots                                   #
#    ##############################################################################
#    LG.info('Start Plots')
#    ## Soundings #################################################################
#    f_cities = f'{here}/soundings.csv'
#    Yt,Xt = np.loadtxt(f_cities,usecols=(0,1),delimiter=',',unpack=True)
#    names = np.loadtxt(f_cities,usecols=(2,),delimiter=',',dtype=str)
#    soundings = [(n,(la,lo))for n,la,lo in zip(names,Yt,Xt)]
#    for place,point in soundings:
#       lat,lon = point
#       if not (left<lon<right and bottom<lat<top): continue
#       name = f'{OUT_folder}/{HH}_sounding_{place}.png'
#       title = f"{place.capitalize()}"
#       title += f" {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
#       LG.info(f'Sounding {place}')
#       ut.sounding(lat,lon,lats,lons,date,ncfile,pressure,tc,td,t2m,ua,va,title,fout=name)
# 
#    ## Scalar properties #########################################################
#    # Background plots ###########################################################
#    dpi = 150
#    ## Terrain 
#    fname = f'{OUT_folder}/terrain.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting terrain')
#       fig,ax,orto = PF.terrain_plot(reflat,reflon,left,right,bottom,top)
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted terrain')
#    
#    ## Parallel and meridian
#    fname = f'{OUT_folder}/meridian.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting meridians')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.parallel_and_meridian(fig,ax,orto,left,right,bottom,top)
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted meridians')
#    
#    ## Rivers
#    fname = f'{OUT_folder}/rivers.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting rivers')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.rivers_plot(fig,ax,orto)
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted rivers')
#    
#    ## CCAA
#    fname = f'{OUT_folder}/ccaa.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting ccaa')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.ccaa_plot(fig,ax,orto)
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted ccaa')
#    
#    ## Cities
#    fname = f'{OUT_folder}/cities.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting cities')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.csv_plot(fig,ax,orto,f'{here}/cities.csv')
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted cities')
#    
#    ## Citiy Names
#    fname = f'{OUT_folder}/cities_names.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting cities names')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.csv_names_plot(fig,ax,orto,f'{here}/cities.csv')
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted cities names')
#    
#    ## Takeoffs 
#    fname = f'{OUT_folder}/takeoffs.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting takeoffs')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.csv_plot(fig,ax,orto,f'{here}/takeoffs.csv')
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted takeoffs')
#    
#    ## Takeoffs Names
#    fname = f'{OUT_folder}/takeoffs_names.png'
#    if os.path.isfile(fname):
#       LG.info(f'{fname} already present')
#    else:
#       LG.debug('plotting takeoffs names')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       PF.csv_names_plot(fig,ax,orto,f'{here}/takeoffs.csv')
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info('plotted takeoffs names')
#    
#    
#    # Properties #################################################################
#    wrf_properties = {'sfcwind':wspd10, 'blwind':blwind, 'bltopwind':bltopwind,
#                      'hglider':hglider, 'wstar':wstar, 'zsfclcl':zsfclcl,
#                      'zblcl':zblcl, 'cape':MCAPE, 'wblmaxmin':wblmaxmin,
#                      'bldepth': bldepth,  #'bsratio':bsratio,
#                      'rain':rain, 'blcloudpct':blcloudpct, 'tdif':tdif,
#                      'lowfrac':low_cloudfrac, 'midfrac':mid_cloudfrac,
#                      'highfrac':high_cloudfrac}
#    
#    colormaps = {'WindSpeed': WindSpeed, 'Convergencias':Convergencias,
#          'greys':greys, 'reds': reds, 'greens':greens, 'blues':blues,
#          'CAPE': CAPE, 'Rain': Rain, 'None':None}
#    
#    titles = {'sfcwind':'Viento Superficie', 'blwind':'Viento Promedio',
#              'bltopwind':'Viento Altura', 'hglider':'Techo (azul)',
#              'wstar':'Térmica', 'zsfclcl':'Base nube', 'zblcl':'Cielo cubierto',
#              'cape':'CAPE', 'wblmaxmin':'Convergencias',
#              'bldepth': 'Altura Capa Convectiva', 'bsratio': 'B/S ratio',
#              'rain': 'Lluvia', 'blcloudpct':'Nubosidad (%)',
#              'tdif': 'Prob. Térmica', 'lowfrac':'Nubosidad baja (%)',
#              'midfrac': 'Nubosidad media (%)', 'highfrac': 'Nubosidad alta (%)'}
#    
#    # plot scalars ###############################################################
#    ftitles = open(f'{OUT_folder}/titles.txt','w')
#    props = ['sfcwind', 'blwind', 'bltopwind', 'wblmaxmin', 'hglider', 'wstar',
#             'bldepth', 'cape', 'zsfclcl', 'zblcl', 'tdif', 'rain',
#             'blcloudpct', 'lowfrac', 'midfrac', 'highfrac']
#    for prop in props:
#       LG.debug(f'plotting {prop}')
#       factor,vmin,vmax,delta,levels,cmap,units = get_properties('plots.ini', prop)
#       try: cmap = colormaps[cmap]
#       except: pass  # XXX cmap is already a cmap name
#       title = titles[prop]
#       title = f"{title} {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
#       M = wrf_properties[prop]
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       C = PF.scalar_plot(fig,ax,orto, lons,lats,wrf_properties[prop]*factor,
#                          delta,vmin,vmax,cmap, levels=levels,
#                          inset_label=date_label)
#       fname = f'{OUT_folder}/{HH}_{prop}.png'
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info(f'plotted {prop}')
#    
#       fname = f'{OUT_folder}/{prop}'
#       if os.path.isfile(fname):
#          LG.info(f'{fname} already present')
#       else:
#          LG.debug('plotting colorbar')
#          ftitles.write(f"{fname} ; {title}\n")
#          PF.plot_colorbar(cmap,delta,vmin,vmax, levels, name=fname,units=units,
#                                 fs=15,norm=None,extend='max')
#          LG.info('plotted colorbar')
#    ftitles.close()
#    
#    ## Vector properties #########################################################
#    names = ['sfcwind','blwind','bltopwind']
#    winds = [[ua10.values, va10.values],
#             [ublavgwind, vblavgwind],
#             [utop, vtop]]
#    
#    for wind,name in zip(winds,names):
#       LG.debug(f'Plotting vector {name}')
#       fig,ax,orto = PF.setup_plot(reflat,reflon,left,right,bottom,top)
#       U = wind[0]
#       V = wind[1]
#       PF.vector_plot(fig,ax,orto,lons.values,lats.values,U,V, dens=1.5,color=(0,0,0))
#       # fname = OUT_folder +'/'+ prefix + name + '_vec.png'
#       fname = f'{OUT_folder}/{HH}_{name}_vec.png'
#       PF.save_figure(fig,fname,dpi=dpi)
#       LG.info(f'Plotted vector {name}')
#    
#    ##XXX shouldn't do this here
#    #wrfout_folder += '/processed'   #gfs_batch.strftime('/%Y/%m/%d/%H')
#    #com = f'mkdir -p {wrfout_folder}'
#    #print('****')
#    #print(com)
#    #os.system(com)
#    #com = f'mv {INfname} {wrfout_folder}'
#    #print('****')
#    #print(com)
#    #os.system(com)

if __name__ == '__main__':

   import sys
   try: INfname = sys.argv[1]
   except IndexError:
      print('File not specified')
      exit()
   
   DOMAIN = get_domain(INfname)

   ################################# LOGGING ####################################
   import logging
   import log_help
   log_file = here+'/'+'.'.join( __file__.split('/')[-1].split('.')[:-1] ) 
   log_file = log_file + f'_{DOMAIN}.log'
   lv = logging.INFO
   logging.basicConfig(level=lv,
                    format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
                    datefmt='%Y/%m/%d-%H:%M:%S',
                    filename = log_file, filemode='a')
   LG = logging.getLogger('main')
   if not is_cron: log_help.screen_handler(LG, lv=lv)
   LG.info(f'Starting: {__file__}')
   ##############################################################################

   ## Output folder
   #XXX should be in a config file
   OUT_folder = '../../Documents/storage/PLOTS/Spain6_1'
   post_process_file(INfname, OUT_folder)
