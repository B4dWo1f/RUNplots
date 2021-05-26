#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
is_cron = bool( os.getenv('RUN_BY_CRON') )
import datetime as dt
import wrf
import metpy.calc as mpcalc
from metpy.units import units
from netCDF4 import Dataset
import wrf_calcs
import plots
import log_help
import logging
LG = logging.getLogger('main')

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))


def get_sounding(date0, lat0, lon0, data_fol, OUT_fol,
                 place='', dom='d02', fout=None):
   """
   This function creates a skew-T (sounding) plot for a given time and place
   date0: requested date for the sounding
   lat0,lon0: requested GPS coordinates for the sounding
   data_fol: path to the wrfout files
   OUT_fol: path to save the plot
   place: name to be used both in the title and in the file name if fout were
          not provided
   dom: TO BE REPLACED. Should be automatic. Domain to look for the WRF data
   fout: file name to save the plot
   """
   fmt_wrfout = '%Y-%m-%d_%H'

   if len(place) == 0:
      place = f'{lat:.3f}, {lon:.3f}'
      LG.debug(f'place not provided. Using: {place}')

   INfname = f'{data_fol}/wrfout_{dom}_{date0.strftime(fmt_wrfout)}:00:00'
   LG.debug('data file: {INfname}')

   # print('File:',INfname)
   ncfile = Dataset(INfname)

   date,lats,lons,terrain,pressure,heights,tc,td,t2m,td2m,ua,va = wrf_calcs.extract.sounding(ncfile)

   HH = date.strftime('%H%M')
   if fout == None:
      fout = f'{OUT_folder}/{HH}_sounding_{place}.png'
      LG.debug(f'fout not provided. Using: {fout}')
   else: fout = fout
   title = f"{place.capitalize()}"
   title += f" {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
   LG.debug(f'Title: {title}')

   lat,lon,p,tc,tdc,t0,u,v,gnd,cu_base_p,cu_base_m,cu_base_t, Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof = wrf_calcs.post_process.sounding(ncfile,lat0,lon0,pressure,tc,td,t2m,ua,va, terrain,lats,lons)
   latlon = f'({lat:.3f},{lon:.3f})'
   plots.sounding.skewt_plot(p,tc,tdc,t0,date,u,v,gnd,
                             cu_base_p,cu_base_m,cu_base_t,
                             Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof,
                             fout=fout,latlon=latlon,title=title)
   return fout


if __name__ == '__main__':
   ################################# LOGGING ####################################
   import logging
   import log_help
   log_file = here+'/'+'.'.join( __file__.split('/')[-1].split('.')[:-1] ) 
   log_file = log_file + f'.log'
   lv = logging.INFO
   logging.basicConfig(level=lv,
                    format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
                    datefmt='%Y/%m/%d-%H:%M:%S',
                    filename = log_file, filemode='w')
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
   LG.info(f'Saved in: {fname}')
