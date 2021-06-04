#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from random import random
from time import sleep
sleep(10*random())

import common
from os.path import expanduser
import wrf_calcs.util as ut

import sys
try: fname = sys.argv[1]
except IndexError:
   print('File not specified')
   exit()

domain = ut.get_domain(fname)
date = common.file2date(fname)

import os
here = os.path.dirname(os.path.realpath(__file__))
is_cron = bool( os.getenv('RUN_BY_CRON') )
################################# LOGGING ####################################
import logging
import log_help
log_file = here+'/'+'.'.join( __file__.split('/')[-1].split('.')[:-1] ) 
log_file = log_file + f'_{domain}.log'
lv = logging.INFO
logging.basicConfig(level=lv,
                 format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
                 datefmt='%Y/%m/%d-%H:%M:%S',
                 filename = log_file, filemode='w')
LG = logging.getLogger('main')
if not is_cron: log_help.screen_handler(LG, lv=lv)
LG.info(f'Starting: {__file__}')
##############################################################################



P = common.get_config()
output_folder = expanduser( P['system']['output_folder'] )
plots_folder = expanduser( P['system']['plots_folder'] )
data_folder = expanduser( P['system']['data_folder'] )
ut.check_directory(output_folder,True)
ut.check_directory(plots_folder,False)
ut.check_directory(data_folder,False)

zooms = common.get_zooms('zooms.ini')

# Read data
A = common.CalcData(fname, plots_folder, data_folder)

# Plots for web
A.plot_background(force=False,zooms=zooms)
A.plot_web(zooms=zooms)


# Plot soundings
with open(f'soundings_{domain}.csv','r') as f:
   for line in f.read().strip().splitlines():
      lat,lon,place = line.split(',')
      lat = float(lat)
      lon = float(lon)
      A.sounding(date,lat,lon,place=place)

LG.info('Done!')
