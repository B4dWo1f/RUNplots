#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
is_cron = bool( os.getenv('RUN_BY_CRON') )
# import util as ut
import datetime as dt
import wrf
import metpy.calc as mpcalc
from metpy.units import units
from netCDF4 import Dataset
import plots
import log_help
import logging
LG = logging.getLogger('main')

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))


from wrf_calcs.extract import getvar
import wrf_calcs
def extract_sounding(ncfile,date0,lat0,lon0):
   # Lats, Lons
   lats = getvar(ncfile, "lat")
   lons = getvar(ncfile, "lon")
   # Date in UTC
   # prefix to save files
   date = str(getvar(ncfile, 'times').values)
   bounds = wrf.geo_bounds(wrfin=ncfile)
   # useful to setup the extent of the maps
   left   = bounds.bottom_left.lon
   right  = bounds.top_right.lon
   bottom = bounds.bottom_left.lat
   top    = bounds.top_right.lat

   if (not date == date0) or\
      (not left < lon0 < right) or\
      (not bottom < lat0 < top):
      print(date,date0, date == date0)
      print(type(date),type(date0))
      print(left,lon0,right,left < lon0 < right)
      print(bottom,lat0,top,bottom < lat0 < top)
      print('Error selecting wrfout file!!')
   else:
      print('Correct wrfout')
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = getvar(ncfile, "pressure")
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   # Also called Geopotential Heights. heights[0,:,:] is the first level, 15m agl   # XXX why 15 and not 10?
   heights = getvar(ncfile, "height")
   # Terrain topography used in the calculations_____________________[m] (ny,nx)
   terrain = getvar(ncfile, "ter")
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = getvar(ncfile, "tc")
   # print('tc',tc.shape)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = getvar(ncfile, "td")
   # print('td',td.shape)
   # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
   t2m = getvar(ncfile, "T2")
   t2m = wrf_calcs.extract.convert_units(t2m, 'degC')
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = getvar(ncfile, "ua")  # U wind component
   va = getvar(ncfile, "va")  # V wind component
   return date,lats,lons,terrain,pressure,heights,tc,td,t2m,ua,va


def get_sounding(date0, lat0, lon0, data_fol, OUT_fol,
                 place='', dom='d02', fout=None):
   fmt_wrfout = '%Y-%m-%d_%H'

   if len(place) == 0:
      place = f'{lat:.3f}, {lon:.3f}'

   # print('File:')
   INfname = f'{data_fol}/wrfout_{dom}_{date0.strftime(fmt_wrfout)}:00:00'

   # print('File:',INfname)
   ncfile = Dataset(INfname)

#    # Lats, Lons
#    lats = wrf.getvar(ncfile, "lat")
#    lons = wrf.getvar(ncfile, "lon")
#    # Date in UTC
#    # prefix to save files
#    date = str(wrf.getvar(ncfile, 'times').values)
#    date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
#    HH = date.strftime('%H%M')
#    bounds = wrf.geo_bounds(wrfin=ncfile)
#    # print('lat/lon',lats.shape)

#    # useful to setup the extent of the maps
#    left   = bounds.bottom_left.lon
#    right  = bounds.top_right.lon
#    bottom = bounds.bottom_left.lat
#    top    = bounds.top_right.lat
#    # print('Forecast for:',date)
#    # print(date0)
#    if (not date == date0) or\
#       (not left < lon0 < right) or\
#       (not bottom < lat0 < top):
#       print('Error selecting wrfout file!!')
#    else:
#       print('Correct wrfout')


#    # Pressure___________________________________________________[hPa] (nz,ny,nx)
#    pressure = wrf.getvar(ncfile, "pressure")
#    # print('Pressure:',pressure.shape)

#    # Temperature_________________________________________________[°C] (nz,ny,nx)
#    tc = wrf.getvar(ncfile, "tc")
#    # print('tc',tc.shape)
#    # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
#    td = wrf.getvar(ncfile, "td", units='degC')
#    # print('td',td.shape)
#    # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
#    t2m = wrf.getvar(ncfile, "T2").metpy.convert_units('degC')
#    t2m.attrs['units'] = 'degC'
#    # print('t2m',t2m.shape)

#    # Wind_______________________________________________________[m/s] (nz,ny,nx)
#    ua = wrf.getvar(ncfile, "ua")  # U wind component
#    va = wrf.getvar(ncfile, "va")  # V wind component
#    wa = wrf.getvar(ncfile, "wa")  # W wind component
   date,lats,lons,terrain,pressure,heights,tc,td,t2m,ua,va = extract_sounding(ncfile,date0,lat0,lon0)


   date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
   HH = date.strftime('%H%M')
   if fout == None: name = f'{OUT_folder}/{HH}_sounding_{place}.png'
   else: name = fout
   title = f"{place.capitalize()}"
   title += f" {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
   # ut.sounding(lat,lon, lats,lons, date, ncfile,
   #             pressure, tc, td, t2m,
   #             ua, va,
   #             title, fout=name)
#sounding(lat,lon,lats,lons,date,ncfile,pressure,tc,td,t0,ua,va,title='',fout='sounding.png')
   i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
   # Get sounding data for specific location
   # h = heights[:,i,j]
   latlon = f'({lats[j,i].values:.3f},{lons[j,i].values:.3f})'
   nk,nj,ni = pressure.shape
   # Make unit aware
   p = pressure[:,j,i].metpy.quantify()
   tc = tc[:,j,i].metpy.quantify()
   tdc = td[:,j,i].metpy.quantify()
   t0 = t2m[j,i].metpy.quantify()
   u = ua[:,j,i] # .metpy.quantify()
   v = va[:,j,i] # .metpy.quantify()
   u = wrf_calcs.extract.convert_units(u, 'km/hour').metpy.quantify()
   v = wrf_calcs.extract.convert_units(v, 'km/hour').metpy.quantify()
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
   cu_base_p, cu_base_t = wrf_calcs.post_process.get_cloud_base(parcel_prof, p, tc, lcl_p, lcl_t)
   cu_base_m = mpcalc.pressure_to_height_std(cu_base_p)
   cu_base_m = cu_base_m.to('m')
   # top
   cu_top_p, cu_top_t = wrf_calcs.post_process.get_cloud_base(parcel_prof, p, tc)
   cu_top_m = mpcalc.pressure_to_height_std(cu_top_p)
   cu_top_m = cu_top_m.to('m')
   # Cumulus matrix
   ps, overcast, cumulus = wrf_calcs.post_process.get_cloud_extension(p,tc,tdc, cu_base_p,cu_top_p)
   rep = 3
   mats =  [overcast for _ in range(rep)]
   mats += [cumulus for _ in range(rep)]
   cloud = np.vstack(mats).transpose()
   Xcloud = np.vstack([range(2*rep) for _ in range(cloud.shape[0])])
   Ycloud = np.vstack([ps for _ in range(2*rep)]).transpose()
   # print('****')
   # print(Xcloud.shape)
   # print(Ycloud.shape)
   # print(overcast.shape)
   # print(cumulus.shape)
   # print(cloud.shape)
   # print('****')
   # exit()
   LG.info('calling skewt plot')
   plots.sounding.skewt_plot(p,tc,tdc,t0,date,u,v,gnd,cu_base_p,cu_base_m,cu_base_t,
         Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof,
         fout=fout,latlon=latlon,title=title)
   return name


if __name__ == '__main__':
   ################################# LOGGING ####################################
   import logging
   import log_help
   log_file = here+'/'+'.'.join( __file__.split('/')[-1].split('.')[:-1] ) 
   log_file = log_file + f'.log'
   lv = logging.DEBUG
   logging.basicConfig(level=lv,
                    format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
                    datefmt='%Y/%m/%d-%H:%M:%S',
                    filename = log_file, filemode='a')
   LG = logging.getLogger('main')
   if not is_cron: log_help.screen_handler(LG, lv=lv)
   LG.info(f'Starting: {__file__}')
   ##############################################################################

   import numpy as np
   from random import randint, choice
   d = choice([17,18,19,20,22])
   # d = 19
   date_req = dt.datetime(2021,5,d,12)   #XXX UTC
   # from random import choice
   # lat,lon = choice( [(40.1,-3.5), (41.17, -3.624)] )
   lat,lon = 41.078854,-3.707029 # arcones ladera
   fname = 'soundings_d02.csv'
   Points = np.loadtxt(fname,usecols=(0,1),delimiter=',')
   names = np.loadtxt(fname,usecols=(2,),delimiter=',',dtype=str)

   ind = randint(0,Points.shape[0]-1)
   # ind = 2
   lat,lon = Points[ind,:]
   place = names[ind]

   # lat,lon = 41.17, -3.624
   data_folder = '../../Documents/storage/WRFOUT/Spain6_1'
   OUT_folder = 'plots'
   fout = 'sounding.png'
   dom = 'd02'
   fname = get_sounding(date_req, lat, lon, data_folder, OUT_folder, place, dom, fout)
   print('Saved in:',fname)
