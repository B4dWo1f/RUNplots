#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
import log_help
import logging
LG = logging.getLogger(__name__)
LG.setLevel(logging.DEBUG)
import wrf_calcs
import plots
import datetime as dt

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()))

class CalcData(object):
   """
   This class encompases all the data we know how to extract from a provided
   wrfout file
   """
   def __init__(self,fname,OUT_folder='.',use_cache=True,read_all=False):
      """
      fname: [str] path to wrfout file
      OUT_folder: [str] path to save processing outputs (property layers,
                        soundings, meteograms...)
      use_cache: [bool] whether or not to use WRF-cache to speed-up
                        variables extraction
      read_all: [bool] extract all available data. If False, each calculation
                       will check for data availability
      """
      self.fname = fname
      self.OUT_folder = OUT_folder
      self.ncfile,\
      self.domain, \
      self.bounds,\
      self.reflat, \
      self.reflon,\
      self.wrfdata,\
      self.date,   \
      self.GFS_batch,\
      self.creation_date = wrf_calcs.extract.read_wrfout_info(fname)
      if use_cache:
         self.cache = wrf_calcs.extract.get_cache(self.ncfile)
         LG.debug('Using cache')
      else:
         self.cache = None
         LG.warning('Not using cache')
      if read_all:
         LG.critical('Not implemented yet')
         #TODO this should call all the methods to retrieve all the variables
   def is_point_in(self,lat,lon):
      """
      Check if coordinates (lat,lon) are within the boundaries of the domain
      """
      left   = self.bounds.bottom_left.lon
      right  = self.bounds.top_right.lon
      bottom = self.bounds.bottom_left.lat
      top    = self.bounds.top_right.lat
      isin = (left < lon < right)  and ( bottom < lat < top)
      if isin: LG.debug(f'Point ({lat},{lon}) is inside domain {self.domain}')
      else:
         LG.warning(f'Point ({lat},{lon}) is NOT inside domain {self.domain}')
         LG.critical('Error selecting wrfout file!!')
         LG.debug(left,lon,right,left < lon < right)
         LG.debug(bottom,lat,top,bottom < lat < top)
      return isin
   def get_sounding_data(self):
      """
      Incorporates the sounding variables as attributes
      """
      LG.warning('Extracting sounding variables')
      date,\
      self.lats, self.lons, self.terrain,\
      self.pressure, self.heights,\
      self.tc, self.td, self.t2m,\
      self.ua, self.va = wrf_calcs.extract.sounding(self.ncfile)
      if not (self.date == date):
         LG.critical('Error with the dates')
   def sounding(self, date0, lat0, lon0,place='',fout=None):
      """
      Plots sounding for date0 and coordinates (lat0,lon0)
      date0: [datetime] date for the requested sounding
      lat0,lon0: [float] coordinates for the requested sounding
      place: [str] label to be used in the title and file name (unless
                   overriden by fout)
      fout: [str] path to save sounding
      """
      if date0 != self.date:
         LG.critical(f'Requested data for {date0} but this file is for {self.date}')
         exit()
      # Check that all variables exist
      try:
         self.ncfile
         self.pressure
         self.tc
         self.td
         self.t2m
         self.ua
         self.va
         self.terrain
         self.lats
         self.lons
      except AttributeError: self.get_sounding_data()

      lat, lon, p,\
      tc, tdc, t0,\
      u, v,\
      gnd,\
      cu_base_p, cu_base_m, cu_base_t,\
      Xcloud,Ycloud,cloud,\
      lcl_p, lcl_t,\
      parcel_prof = wrf_calcs.post_process.sounding(self.ncfile, lat0, lon0,
                                                    self.pressure,
                                                    self.tc, self.td, self.t2m,
                                                    self.ua,self.va,
                                                    self.terrain,
                                                    self.lats,self.lons)
      # Settings for saving image
      latlon = f'({lat:.3f},{lon:.3f})'
      HH = self.date.strftime('%H%M')
      if len(place) == 0:
         place = f'{lat0:.3f},{lon0:.3f}'
         LG.debug(f'place not provided. Using: {place}')
      if fout == None:
         fout = f'{self.OUT_folder}/{HH}_sounding_{place}.png'
         LG.debug(f'fout not provided. Using: {fout}')
      else: LG.debug(f'fout was provided by user')
      title = f"{place.capitalize()}"
      title += f" {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
      LG.debug(f'Title: {title}')

      plots.sounding.skewt_plot(p,tc,tdc,t0,date,u,v,gnd,
                                cu_base_p,cu_base_m,cu_base_t,
                                Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof,
                                fout=fout,latlon=latlon,title=title)

   def __str__(self):
      txt =  f'Data from: {self.wrfdata}\n'
      txt += f'Domain: {self.domain}\n'
      txt += f'Ref lat/lon: {self.reflat}/{self.reflon}\n'
      txt += f'GFS batch: {self.GFS_batch}\n'
      return txt


if __name__ == '__main__':
   import datetime as dt
   fmt =  '%Y-%m-%d_%H:%M:%S'

   domain = 'd02'
   date = dt.datetime(2021,5,22,12)

   folder = f'../../Documents/storage/WRFOUT/Spain6_1/'
   fname  = f'{folder}/wrfout_{domain}_{date.strftime(fmt)}'

   A = CalcData(fname)

   with open(f'soundings_{domain}.csv','r') as f:
      soundings = f.read().strip().splitlines()

   for line in soundings:
      lat,lon,place = line.split(',')
      lat = float(lat)
      lon = float(lon)
      A.sounding(date,lat,lon,place=place)
