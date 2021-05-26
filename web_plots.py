#!/usr/bin/python3
# -*- coding: UTF-8 -*-

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
ut.check_directory(output_folder,True)
ut.check_directory(plots_folder,False)

A = common.CalcData(fname, plots_folder)

A.plot_background(False)
A.plot_web()

with open(f'soundings_{domain}.csv','r') as f:
   soundings = f.read().strip().splitlines()

for line in soundings:
   lat,lon,place = line.split(',')
   lat = float(lat)
   lon = float(lon)
   A.sounding(date,lat,lon,place=place)
LG.info('Done!')
