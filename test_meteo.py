#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import json
import common
import datetime as dt
import numpy as np
from os.path import expanduser
import wrf_calcs.util as ut
import plots.geography as geo

P = common.get_config()
output_folder = expanduser( P['system']['output_folder'] )
plots_folder = expanduser( P['system']['plots_folder'] )
data_folder = expanduser( P['system']['data_folder'] )
ut.check_directory(output_folder,stop=True)  # Stop since WRFOUT doen't exist
ut.check_directory(plots_folder,stop=False)
ut.check_directory(data_folder,stop=False)


folder = data_folder
date = dt.datetime(2021,5,28,12)
lat,lon,place = 41.078854,-3.707029,'arcones'




####################
folder = f"{data_folder}/{date.strftime('%Y/%m/%d')}"

import wrf
from netCDF4 import Dataset
from time import time

def projector(ncfile):
   def aux(lat,lon):
      return wrf.ll_to_xy(ncfile, lat, lon).values
   return aux
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))

hours = range(8,21)

for h in hours:
   date_h = date.replace(hour=h)-UTCshift
   fname = f"{output_folder}/wrfout_d02_{date_h.strftime('%Y-%m-%d_%H:%M:%S')}"
   ncfile = Dataset(fname)
   proj = projector(ncfile)
   # Borders
   borders =  f"{folder}/{date.strftime('%H%M')}_borders.json"
   with open(borders, 'r') as json_file:
       borders = json.load(json_file)

   reflat = borders['reflat']
   reflon = borders['reflon']
   left = borders['bot_left']['lon']
   right = borders['top_right']['lon']
   bottom = borders['bot_left']['lat']
   top = borders['top_right']['lat']

   # Read lat/lon
   lats = f"{folder}/{date.strftime('%H%M')}_lats.npy"
   lons = f"{folder}/{date.strftime('%H%M')}_lons.npy"
   lats = np.load(lats)
   lons = np.load(lons)
   with open(f"{folder}/{date.strftime('%H%M')}_label.txt", 'r') as fp:
      date_label = fp.read().strip()

   scalars = ['u', 'v','wstar','hcrit','heights','t', 'td'] #, 'clouds']
   for scalar in scalars:
      title = scalar
      scalar = f"{folder}/{date.strftime('%H%M')}_{scalar}.npy"
      scalar = np.load(scalar)

