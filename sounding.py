#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# import os
# here = os.path.dirname(os.path.realpath(__file__))
import util as ut
import datetime as dt
import wrf
from netCDF4 import Dataset

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))


def get_sounding(date0, lat0, lon0, data_fol, OUT_fol,
                 place='', dom='d02', fout=None):
   fmt_wrfout = '%Y-%m-%d_%H'

   if len(place) == 0:
      place = f'{lat:.3f}, {lon:.3f}'

   # print('File:')
   INfname = f'{data_fol}/wrfout_{dom}_{date0.strftime(fmt_wrfout)}:00:00'


   # print('File:',INfname)
   ncfile = Dataset(INfname)

   # Lats, Lons
   lats = wrf.getvar(ncfile, "lat")
   lons = wrf.getvar(ncfile, "lon")
   # Date in UTC
   # prefix to save files
   date = str(wrf.getvar(ncfile, 'times').values)
   date = dt.datetime.strptime(date[:-3], '%Y-%m-%dT%H:%M:%S.%f')
   HH = date.strftime('%H%M')
   bounds = wrf.geo_bounds(wrfin=ncfile)
   # print('lat/lon',lats.shape)

   # useful to setup the extent of the maps
   left   = bounds.bottom_left.lon
   right  = bounds.top_right.lon
   bottom = bounds.bottom_left.lat
   top    = bounds.top_right.lat
   # print('Forecast for:',date)
   # print(date0)
   if (not date == date0) or\
      (not left < lon0 < right) or\
      (not bottom < lat0 < top):
      print('Error selecting wrfout file!!')
   else:
      print('Correct wrfout')


   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = wrf.getvar(ncfile, "pressure")
   # print('Pressure:',pressure.shape)

   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = wrf.getvar(ncfile, "tc")
   # print('tc',tc.shape)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = wrf.getvar(ncfile, "td", units='degC')
   # print('td',td.shape)
   # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
   t2m = wrf.getvar(ncfile, "T2").metpy.convert_units('degC')
   t2m.attrs['units'] = 'degC'
   # print('t2m',t2m.shape)

   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = wrf.getvar(ncfile, "ua")  # U wind component
   va = wrf.getvar(ncfile, "va")  # V wind component
   wa = wrf.getvar(ncfile, "wa")  # W wind component


   if fout == None: name = f'{OUT_folder}/{HH}_sounding_{place}.png'
   else: name = fout
   title = f"{place.capitalize()}"
   title += f" {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
   ut.sounding(lat,lon, lats,lons, date, ncfile,
               pressure, tc, td, t2m,
               ua, va,
               title, fout=name)
   return name


if __name__ == '__main__':
   import numpy as np
   from random import randint
   date_req = dt.datetime(2021,5,17,14)   #XXX UTC
   # from random import choice
   # lat,lon = choice( [(40.1,-3.5), (41.17, -3.624)] )
   lat,lon = 41.078854,-3.707029
   fname = 'soundings.csv'
   Points = np.loadtxt(fname,usecols=(0,1),delimiter=',')
   names = np.loadtxt(fname,usecols=(2,),delimiter=',',dtype=str)

   ind = randint(0,Points.shape[0]-1)
   lat,lon = Points[ind,:]
   place = names[ind]

   # lat,lon = 41.17, -3.624
   data_folder = '../../Documents/storage/WRFOUT/Spain6_1/'
   OUT_folder = 'plots'
   fout = 'sounding.png'
   dom = 'd02'
   fname = get_sounding(date_req, lat, lon, data_folder, OUT_folder, place, dom, fout)
   print('Saved in:',fname)
