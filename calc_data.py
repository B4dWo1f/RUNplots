#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
HOME = os.getenv('HOME')
import datetime as dt
import utils as ut
import extract_wrf as ex

class CalcData(object):
   """
   This class encompases all the data we know how to extract from a provided
   wrfout file
   """
   def __init__(self, fname, OUT_folder='.', DATA_folder='.'):
      """
      fname: [str] path to wrfout file
      OUT_folder: [str] path to save processing outputs (property layers,
                        soundings, meteograms...)
      self.prevnc: [netCDF4.Dataset] with previous hour for plotting rain
      """
      self.fname = fname
      self.dpi = 90
      if not os.path.isfile(fname):
         # LG.critical(f'FileNotFound: {fname}')
         print(f'FILE DOES NOT EXIST: {fname}')
         exit()
      # LG.info(fname)
      info = ex.wrfout_info(fname)
      for key, value in info.items():
         setattr(self, key, value)
      # date-based nameing like: file_{self.tail}.ext
      self.tail_d = self.date.strftime('%Y%m%d')
      self.tail_h = self.date.strftime('%H%M')
      self.tail = f'{self.tail_d}_{self.tail_h}'
      # Previous file
      fname_folder = '/'.join(self.fname.split('/')[:-1])
      for prev_folder in [fname_folder, fname_folder+'/processed']:
         try:
            h1 = dt.timedelta(hours=1)
            prev = ut.date2file(self.date - h1,self.domain,prev_folder)
            # LG.debug(f'Looking for previous wrfout in {prev_folder}')
            info = ex.wrfout_info(prev)
            self.prevnc = info['ncfile']
            # LG.debug(f'Found it!')
            break
         except FileNotFoundError:
            self.prevnc = None
            # LG.warning('Previous wrfout not found')

      # Fix Out Folder
      path = [OUT_folder,self.domain,self.date.strftime('%Y/%m/%d')]
      self.OUT_folder = '/'.join(path)
#      ut.check_directory(self.OUT_folder)
      # Fix Data Folder
      # LG.debug(f'DATA: {DATA_folder}')
      path = [DATA_folder,self.domain,self.date.strftime('%Y/%m/%d')]
      self.DATA_folder = '/'.join(path)
#      ut.check_directory(self.DATA_folder)
      # LG.info(f'DATA folder: {DATA_folder}')
      # Borders
      self.left   = self.bounds.bottom_left.lon
      self.right  = self.bounds.top_right.lon
      self.bottom = self.bounds.bottom_left.lat
      self.top    = self.bounds.top_right.lat
      self.borders = [self.left, self.right, self.bottom, self.top]
      self.cache = {} # Cache will be populated as needed
      # Extract WRF data
      wrf_vars = ex.wrf_vars(self.ncfile, prevnc=self.prevnc, cache=self.cache)
      self.wrf_vars = wrf_vars
      # for k,v in self.wrf_vars.items():
      #    print(f'{k:<15} {v.shape}')

      drjack_vars = ex.drjack_vars(self.wrf_vars)
      # u,v,w, hfx, pressure,heights, terrain, bldepth,tc, td,qvapor )
      self.drjack_vars = drjack_vars
      # for k,v in self.drjack_vars.items():
      #    print(f'{k:<15} {v.shape}')
      ############# Sanity check ############
      aux = []
      for k,v in self.wrf_vars.items():
         aux.append(v.shape[-2:])
      for k,v in self.drjack_vars.items():
         aux.append(v.shape[-2:])
      if (len(set(aux))) > 1: raise
      #######################################
   def __str__(self):
      txt =  f'WRFout folder: {self.wrfout_folder}\n'
      txt += f' Plots folder: {self.OUT_folder}\n'
      txt += f'  Data folder: {self.DATA_folder}\n'
      txt += f'Domain: {self.domain}\n'
      txt += f'Ref lat/lon: {self.reflat}/{self.reflon}\n'
      txt += f"File creation {self.creation_date}\n"
      txt += f'Forecast date: {self.date}\n'
      txt += f'    GFS batch: {self.GFS_batch}\n'
      txt += f'\n========= WRF Variables =========\n'
      for k,v in self.wrf_vars.items():
         txt += ut.pretty_print_var(v)
      txt += f"\n========= DrJack's derived Variables =========\n"
      for k,v in self.drjack_vars.items():
         txt += ut.pretty_print_var(v)
      return txt
