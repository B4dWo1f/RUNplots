#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
import log_help
import logging
LG = logging.getLogger(__name__)
LG.setLevel(logging.DEBUG)

import numpy as np
import wrf_calcs.extract as ex
import wrf_calcs.post_process as post
import wrf_calcs.util as ut
import plots
import plots.geography as geo
import datetime as dt
fmt =  '%Y-%m-%d_%H:%M:%S'

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))

class CalcData(object):
   """
   This class encompases all the data we know how to extract from a provided
   wrfout file
   """
   def __init__(self,fname,OUT_folder='.',use_cache=True,read_all=True):
      """
      fname: [str] path to wrfout file
      OUT_folder: [str] path to save processing outputs (property layers,
                        soundings, meteograms...)
      use_cache: [bool] whether or not to use WRF-cache to speed-up
                        variables extraction
      read_all: [bool] extract all available data. If False, each calculation
                       will check for data availability
      """
      self.dpi=90
      if not os.path.isfile(fname):
         LG.critical(f'FileNotFound: {fname}')
         exit()
      LG.info(fname)
      self.fname = fname
      self.ncfile,\
      self.domain, \
      self.bounds,\
      self.reflat, \
      self.reflon,\
      self.wrfdata,\
      self.date,   \
      self.GFS_batch,\
      self.creation_date = ex.read_wrfout_info(fname)
      # Fix Out Folder
      path = [OUT_folder,self.domain,self.date.strftime('%Y/%m/%d')]
      self.OUT_folder = '/'.join(path)
      ut.check_directory(self.OUT_folder)
      # Borders
      self.left   = self.bounds.bottom_left.lon
      self.right  = self.bounds.top_right.lon
      self.bottom = self.bounds.bottom_left.lat
      self.top    = self.bounds.top_right.lat
      self.borders = [self.left, self.right, self.bottom, self.top]
      if use_cache:
         self.cache = ex.get_cache(self.ncfile)
         LG.debug('Using cache')
      else:
         self.cache = None
         LG.warning('Not using cache')
      if read_all:
         self.readWRF()
         self.derived_quantities()
   def is_point_in(self,lat,lon):
      """
      Check if coordinates (lat,lon) are within the boundaries of the domain
      """
      isin = (self.left < lon < self.right) and (self.bottom < lat < self.top)
      if isin: LG.debug(f'Point ({lat},{lon}) is inside domain {self.domain}')
      else:
         LG.warning(f'Point ({lat},{lon}) is NOT inside domain {self.domain}')
         LG.critical('Error selecting wrfout file!!')
         LG.debug(self.left,lon,self.right,self.left < lon < self.right)
         LG.debug(self.bottom,lat,self.top,self.bottom < lat < self.top)
      return isin
   @log_help.timer(LG)
   def readWRF(self):
      """
      Read all the variables
      lats,lons: [ny,nx] grid of latitudes and longitudes
      u,v,w
      u10,v10
      wspd,wdir,
      wspd10,wdir10
      pressure
      heights
      terrain
      bldepth
      hfx
      qcloud
      qvapor
      tc
      td
      tc2m
      td2m
      CAPE
      rain
      low_cloudfrac, mid_cloudfrac, high_cloudfrac
      blcloudpct
      """
      LG.info('Reading variables')
      self.lats, self.lons,\
      self.u,self.v,self.w, self.u10,self.v10,\
      self.wspd, self.wdir, self.wspd10, self.wdir10,\
      self.pressure, self.heights, self.terrain,\
      self.bldepth, self.pblh, self.hfx, self.qcloud, self.qvapor,\
      self.tc, self.td, self.tc2m, self.td2m, self.tsk,\
      self.CAPE, self.rain,\
      self.low_cloudfrac, self.mid_cloudfrac, self.high_cloudfrac,\
      self.blcloudpct = ex.all_properties(self.ncfile, self.cache)
      LG.info('All variables imported')
   def derived_quantities(self):
      """
      Variables not provided by WRF (nor WRF-python)
      Derived quantities either experimental 
      """
      self.tdif = self.tsk-self.tc2m
      self.wblmaxmin, self.wstar, self.hcrit,\
      self.zsfclcl, self.zblcl, self.hglider,\
      self.ublavgwind, self.vblavgwind, self.blwind,\
      self.utop, self.vtop,\
      self.bltopwind = post.drjacks_vars(self.u,self.v,self.w, self.hfx,
                                         self.pressure,self.heights,self.terrain,
                                         self.bldepth,self.tc, self.td,
                                         self.qvapor)

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
         LG.critical(f'Requested data for {date0} but file is for {self.date}')
         exit()
      # Check that all variables exist
      try:
         self.ncfile
         self.pressure
         self.tc
         self.tc2m
         self.td
         self.td2m
         self.ua
         self.va
         self.terrain
         self.lats
         self.lons
      except AttributeError:
         LG.warning('Sounding variables not available')
         self.get_sounding_data()

      LG.info(f'project variables to ({lat0:.3f},{lon0:.3f})')
      lat, lon, p,\
      tc, tdc, t0, td0,\
      u, v,\
      gnd,\
      cu_base_p, cu_base_m, cu_base_t,\
      Xcloud,Ycloud,cloud,\
      lcl_p, lcl_t,\
      parcel_prof = post.sounding(self.ncfile, lat0, lon0,
                                                    self.pressure,
                                                    self.tc, self.td,
                                                    self.tc2m, self.td2m,
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
      title += f" {(self.date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
      LG.debug(f'Title: {title}')

      LG.info(f'Plotting')
      plots.sounding.skewt_plot(p,tc,tdc,t0,td0,self.date,u,v,gnd,
                                cu_base_p,cu_base_m,cu_base_t,
                                Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof,
                                fout=fout,latlon=latlon,title=title)
   def get_sounding_data(self):
      """
      Incorporates the sounding variables as attributes
      """
      LG.info('Extracting sounding variables')
      date,\
      self.lats, self.lons, self.terrain,\
      self.pressure, self.heights,\
      self.tc, self.td, self.tc2m, self.td2m,\
      self.ua, self.va = ex.sounding(self.ncfile)
      if not (self.date == date):
         LG.critical('Error with the dates')
   def plot_background(self,force=False):
      ## Terrain 
      fname = f'{self.OUT_folder}/terrain.png'
      if not os.path.isfile(fname) and not force:
         LG.debug('plotting terrain')
         fig,ax,orto = geo.terrain(self.reflat,self.reflon,*self.borders)
         geo.save_figure(fig,fname,dpi=self.dpi)
         LG.info('plotted terrain')
      else:
         LG.info(f'{fname} already present')
      ## Parallel and meridian
      fname = f'{self.OUT_folder}/meridian.png'
      if not os.path.isfile(fname) and not force:
         LG.debug('plotting meridians')
         fig,ax,orto = geo.setup_plot(self.reflat,self.reflon,*self.borders)
         plots.geo.parallel_and_meridian(fig,ax,orto,*self.borders)
         plots.geo.save_figure(fig,fname,dpi=self.dpi)
         LG.info('plotted meridians')
      else:
         LG.info(f'{fname} already present')
      file_func = [(f'{self.OUT_folder}/rivers.png',geo.rivers_plot,[]),
                   (f'{self.OUT_folder}/ccaa.png', geo.ccaa_plot,[]),
                   (f'{self.OUT_folder}/cities.png', geo.csv_plot,
                                                [f'{here}/cities.csv']),
                   (f'{self.OUT_folder}/cities_names.png', geo.csv_names_plot,
                                                [f'{here}/cities.csv']),
                   (f'{self.OUT_folder}/takeoffs.png', geo.csv_plot,
                                              [f'{here}/takeoffs.csv']),
                   (f'{self.OUT_folder}/takeoffs_names.png',
                                              geo.csv_names_plot,
                                              [f'{here}/takeoffs.csv']) ]
      for fname,func,args in file_func:
         if not os.path.isfile(fname) and not force:
            LG.debug(f"plotting {fname.split('/')[-1]}")
            fig,ax,orto=geo.setup_plot(self.reflat,self.reflon,*self.borders)
            func(fig,ax,orto,*args)
            plots.geo.save_figure(fig,fname,dpi=self.dpi)
            LG.info(f"plotted {fname.split('/')[-1]}")
         else:
            LG.info(f"{fname.split('/')[-1]} already present")
   def plot_web(self,fname='plots.ini'):
      """
      """
      wrf_properties = {'sfcwind':self.wspd10, 'blwind':self.blwind,
            'bltopwind':self.bltopwind, 'hglider':self.hglider,
            'wstar':self.wstar, 'zsfclcl':self.zsfclcl, 'zblcl':self.zblcl,
            'cape':self.CAPE, 'wblmaxmin':self.wblmaxmin,
            'bldepth':self.bldepth,  #'bsratio':bsratio,
            'rain':self.rain, 'blcloudpct':self.blcloudpct, 'tdif':self.tdif,
            'lowfrac':self.low_cloudfrac, 'midfrac':self.mid_cloudfrac,
            'highfrac':self.high_cloudfrac}
      props = ['sfcwind', 'blwind', 'bltopwind', 'wblmaxmin', 'hglider',
               'wstar', 'bldepth', 'cape', 'zsfclcl', 'zblcl', 'tdif', 'rain',
               'blcloudpct', 'lowfrac', 'midfrac', 'highfrac']
      left   = self.bounds.bottom_left.lon
      right  = self.bounds.top_right.lon
      bottom = self.bounds.bottom_left.lat
      top    = self.bounds.top_right.lat
      HH = self.date.strftime('%H%M')
      date_label = 'valid: ' + self.date.strftime( fmt ) + 'z\n'
      date_label +=  'GFS: ' + self.GFS_batch.strftime( fmt ) + '\n'
      date_label += 'plot: ' + self.creation_date.strftime( fmt+' ' )
      for prop in props:
         LG.info(prop)
         factor,vmin,vmax,delta,levels,cmap,units,title = post.scalar_props('plots.ini', prop)
         title = f"{title} {(self.date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
         fig,ax,orto = plots.geo.setup_plot(self.reflat,self.reflon,
                                            *self.borders)
         C = plots.geo.scalar_plot(fig,ax,orto, self.lons,self.lats,
                                   wrf_properties[prop]*factor,
                            delta,vmin,vmax,cmap, levels=levels,
                            inset_label=date_label)
         fname = f'{self.OUT_folder}/{HH}_{prop}.png'
         plots.geo.save_figure(fig,fname,dpi=self.dpi)
         LG.info(f'plotted {prop}')
      
         fname = f'{self.OUT_folder}/{prop}'
         if os.path.isfile(fname):
            LG.info(f'{fname} already present')
         else:
            LG.debug('plotting colorbar')
            plots.geo.plot_colorbar(cmap,delta,vmin,vmax, levels,
                                    name=fname,units=units,
                                    fs=15,norm=None,extend='max')
            LG.info('plotted colorbar')
      names = ['sfcwind','blwind','bltopwind']
      winds = [[self.u10.values, self.v10.values],
               [self.ublavgwind, self.vblavgwind],
               [self.utop, self.vtop]]
      for wind,name in zip(winds,names):
         LG.debug(f'Plotting vector {name}')
         fig,ax,orto = plots.geo.setup_plot(self.reflat,self.reflon,
                                            *self.borders)
         U = wind[0]
         V = wind[1]
         plots.geo.vector_plot(fig,ax,orto,self.lons.values,self.lats.values,U,V, dens=1.5,color=(0,0,0))
         # fname = OUT_folder +'/'+ prefix + name + '_vec.png'
         fname = f'{self.OUT_folder}/{HH}_{name}_vec.png'
         plots.geo.save_figure(fig,fname,dpi=self.dpi)
         LG.info(f'Plotted vector {name}')




   def __str__(self):
      txt =  f'Data from: {self.wrfdata}\n'
      txt += f'Domain: {self.domain}\n'
      txt += f'Ref lat/lon: {self.reflat}/{self.reflon}\n'
      txt += f'GFS batch: {self.GFS_batch}\n'
      return txt


from configparser import ConfigParser, ExtendedInterpolation
from os.path import expanduser
def get_config(fname='plots.ini'):
   """
   Return the data for plotting property. Intended to read from plots.ini
   """
   LG.info(f'Loading config file: {fname}')
   # if not os.path.isfile(fname): return None
   config = ConfigParser(inline_comment_prefixes='#')
   config._interpolation = ExtendedInterpolation()
   config.read(fname)
   props = config._sections
   for item,value in props.items():
      for k,v in value.items():
         if k in ['factor','delta','vmin','vmax']: value[k] = float(v)
   return props


def date2file(date,domain,folder):
   return f'{folder}/wrfout_{domain}_{date.strftime(fmt)}'

def file2date(fname):
   date = fname.split('/')[-1]
   date = '_'.join(date.split('_')[-2:])
   return dt.datetime.strptime(date,fmt)


if __name__ == '__main__':
   P = get_config()

   domain = 'd02'
   date = dt.datetime(2021,5,22,12)

   folder = f'../../Documents/storage/WRFOUT/Spain6_1'
   fname  = f'{folder}/wrfout_{domain}_{date.strftime(fmt)}'

   import os
   from random import choice
   files = os.popen(f'ls {folder}/*d02*').read().strip().split()

   fname = choice(files)
   date = file2date(fname)

   date = dt.datetime(2021,5,25,15)
   fname = date2file(date,'d02',folder)

   LG.info(fname)
   LG.info(date)

   A = CalcData(fname)

   A.plot_web()

   with open(f'soundings_{domain}.csv','r') as f:
      soundings = f.read().strip().splitlines()

   for line in soundings:
      lat,lon,place = line.split(',')
      lat = float(lat)
      lon = float(lon)
      A.sounding(date,lat,lon,place=place)
