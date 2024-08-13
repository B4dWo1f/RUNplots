#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
from time import sleep
import datetime as dt
################################# LOGGING ####################################
import logging
log_file = '.'.join( __file__.split('/')[-1].split('.')[:-1] ) + '.log'
lv = logging.DEBUG
fmt='%(asctime)s:%(name)s:%(levelname)s: %(message)s'
logging.basicConfig(level=lv, format=fmt, datefmt='%Y/%m/%d-%H:%M:%S',
                              filename = log_file, filemode='w')
LG = logging.getLogger('main')
########## Screen Logger (optional) ##########
sh = logging.StreamHandler()                 #
sh.setLevel(logging.INFO)                    #
fmt = '%(name)s:%(levelname)s: %(message)s'  #
fmt = logging.Formatter(fmt)                 #
sh.setFormatter(fmt)                         #
LG.addHandler(sh)                            #
##############################################################################
LG.info(f'Starting: {__file__}')



def check_files(fol):
    return os.popen(f'ls -l {fol}/wrfout_d02_*_20*').read().strip().splitlines()
    # return os.popen(f'ls -l {fol}/wrfout_d02_*_19*').read().strip().splitlines()


fol = '/storage/WRFOUT/Spain6_1/processed'
files = check_files(fol)

while not os.path.isfile('STOP_meteograms'):
   icont = 0
   while check_files(fol) == files:
      icont += 1
      if icont%30 ==0: LG.debug('No new files')
      sleep(10)

   LG.info('New day to meteogram')
   files_new = check_files(fol)
   difference = list(set(files_new) - set(files))


   for f in difference:
      LG.info(f'plotting file: {f}')
      date = f.split()[-1].split('/')[-1]
      date = dt.datetime.strptime(date,'wrfout_d02_%Y-%m-%d_%H:%M:%S')
      today = dt.datetime.now().date()
      day_shift = int((date.date() - today).total_seconds()/24/60/60)
      LG.debug(f'making meteogram for day {date.date()} ({day_shift})')
      com = f'python3 meteograms.py {day_shift} &'
      LG.debug(com)
      os.system(com)
   files = files_new

LG.info('Exiting watch_meteograms.py. Bye!')
