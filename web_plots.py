#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
This script will plot all the layers shown in the web http://raspuri.mooo.com/
Assumptions inherited from our way to run WRF (mainly file structure):
- wrfout files are outputed to wrfout_folder
- wrfout_folder should contain a batch.txt file containing the batch of GFS
data used for the wrfout files
"""

# WRF and maps
import wrf_calcs
# from netCDF4 import Dataset
# import wrf
# import metpy
# import rasterio
# from rasterio.merge import merge
# # My libraries
# from colormaps import WindSpeed, Convergencias, CAPE, Rain
# from colormaps import greys, reds, greens, blues
# import util as ut
# # import plot_functions as PF   # My plotting functions
import plots
# # Standard libraries
from configparser import ConfigParser, ExtendedInterpolation
from os.path import expanduser
import numpy as np
import os
# import sys
here = os.path.dirname(os.path.realpath(__file__))
is_cron = bool( os.getenv('RUN_BY_CRON') )
import log_help
import logging
LG = logging.getLogger('main')


import datetime as dt
fmt = '%d/%m/%Y-%H:%M'

# Get UTCshift
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))
LG.info(f'UTC shift: {UTCshift}')

# @log_help.timer(LG)
def getvar(ncfile,name,cache=None):
   """
   wrapper for wrf.getvar to include debug messages
   """
   aux = wrf.getvar(ncfile, name, cache=cache)
   try: LG.debug(f'{name}: [{aux.units}] {aux.shape}')
   except: LG.debug(f'{name}: {aux.shape}')
   return aux




# def get_sounding_vars(ncfile,my_cache=None):
#    """
#    Extract the variables required for soundings
#    """
#    # Latitude, longitude___________________________________________[deg] (ny,nx)
#    lats = getvar(ncfile, "lat", cache=my_cache)
#    lons = getvar(ncfile, "lon", cache=my_cache)
#    # Terrain topography used in the calculations_____________________[m] (ny,nx)
#    terrain = getvar(ncfile, "ter", cache=my_cache) # = HGT
#    print(terrain)
#    exit()
#    # Pressure___________________________________________________[hPa] (nz,ny,nx)
#    pressure = getvar(ncfile, "pressure", cache=my_cache)
#    # Temperature_________________________________________________[°C] (nz,ny,nx)
#    tc = getvar(ncfile, "tc", cache=my_cache)
#    # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
#    td = getvar(ncfile, "td", units='degC', cache=my_cache)
#    # Temperature 2m above ground________________________________[K-->°C] (ny,nx)
#    t2m = getvar(ncfile, "T2", cache=my_cache)
#    t2m = convert_units(t2m,'degC')
#    LG.debug(f't2m: [{t2m.units}] {t2m.shape}')
#    # Wind_______________________________________________________[m/s] (nz,ny,nx)
#    ua = getvar(ncfile, "ua", cache=my_cache)  # U wind component
#    va = getvar(ncfile, "va", cache=my_cache)  # V wind component
#    return lats,lons,pressure,tc,td,t2m,terrain,ua,va
# 
# def sounding(lat,lon,lats,lons,date,ncfile,pressure,tc,td,t0,ua,va,
#                                                  title='',fout='sounding.png'):
#    """
#    lat,lon: spatial coordinates for the sounding
#    date: UTC date-time for the sounding
#    ncfile: ntcd4 Dataset from the WRF output
#    tc: Model temperature in celsius
#    tdc: Model dew temperature in celsius
#    t0: Model temperature 2m above ground
#    ua: Model X wind (m/s)
#    va: Model Y wind (m/s)
#    fout: save fig name
#    """
#    LG.info('Starting sounding')
#    i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
#    # Get sounding data for specific location
#    # h = heights[:,i,j]
#    latlon = f'({lats[j,i].values:.3f},{lons[j,i].values:.3f})'
#    nk,nj,ni = pressure.shape
#    p = pressure[:,j,i]
#    tc = tc[:,j,i]
#    tdc = td[:,j,i]
#    u = ua[:,j,i]
#    v = va[:,j,i]
#    t0 = t0[j,i]
#    LG.info('calling skewt plot')
#    PL.skewt_plot(p,tc,tdc,t0,date,u,v,fout=fout,latlon=latlon,title=title)
# 
# 
# def find_cross(left,right,p,tc,interp=True):
#    if interp:
#       ps = np.linspace(np.max(p),np.min(p),500)
#       left = interp1d(p,left)(ps)
#       right = interp1d(p,right)(ps)
#       aux = (np.diff(np.sign(left-right)) != 0)*1
#       ind, = np.where(aux==1)
#       ind_cross = np.min(ind)
#       # ind_cross = np.argmin(np.abs(left-right))
#       p_base = ps[ind_cross]
#       t_base = right[ind_cross]
#    else:
#       aux = (np.diff(np.sign(left-right)) != 0)*1
#       ind, = np.where(aux==1)
#       ind_cross = np.min(ind)
#       p_base = p[ind_cross]
#       t_base = tc[ind_cross]
#    return p_base, t_base
# 
# 
# def get_cloud_base(parcel_profile,p,tc,lcl_p,lcl_t):
#    # Plot cloud base
#    p_base, t_base = find_cross(parcel_prof.magnitude, tc, p, tc, interp=True)
#    p_base = np.max([lcl_p, p_base])
#    t_base = np.max([lcl_t, t_base])
#    return p_base, t_base
# 
# def overcast(p,tc,td):
#    ## Clouds ############
#    ps = np.linspace(np.max(p),np.min(p),500)
#    tcs = interp1d(p,tc)(ps)
#    tds = interp1d(p,tdc)(ps)
#    x0 = 0.3
#    t = 0.2
#    overcast = fermi(tcs-tds, x0=x0,t=t)
#    overcast = overcast/fermi(0, x0=x0,t=t)
#    cumulus = np.where((p_base>ps) & (ps>parcel_cross[0]),1,0)
#    return overcast, cumulus



def get_config(fname='plots.ini'):
   """
   Return the data for plotting property. Intended to read from plots.ini
   """
   LG.info(f'Loading config file: {fname}')
   # if not os.path.isfile(fname): return None
   config = ConfigParser(inline_comment_prefixes='#')
   config._interpolation = ExtendedInterpolation()
   config.read(fname)
   #XXX We shouldn't use eval
   out_folder = expanduser(config['system']['output_folder'])
   return out_folder

#def scalar_props(fname,section):
#   """
#   Return the data for plotting property. Intended to read from plots.ini
#   """
#   LG.info(f'Loading config file: {fname} for section {section}')
#   # if not os.path.isfile(fname): return None
#   config = ConfigParser(inline_comment_prefixes='#')
#   config._interpolation = ExtendedInterpolation()
#   config.read(fname)
#   #XXX We shouldn't use eval
#   factor = float(eval(config[section]['factor']))
#   vmin   = float(eval(config[section]['vmin']))
#   vmax   = float(eval(config[section]['vmax']))
#   delta  = float(eval(config[section]['delta']))
#   try: levels = config.getboolean(section, 'levels')
#   except: levels = config[section].get('levels')
#   if levels == False: levels = None
#   elif levels != None:
#      levels = levels.replace(']','').replace('[','')
#      levels = list(map(float,levels.split(',')))
#      levels = [float(l) for l in levels]
#   else: levels = []
#   cmap = config[section]['cmap']
#   units = config[section]['units']
#   return factor,vmin,vmax,delta,levels,cmap,units
 
 
def post_process_file(INfname,OUT_folder='plots'):
   ncfile,DOMAIN,wrfout_folder,reflat,reflon, date,gfs_batch,creation_date =\
   wrf_calcs.extract.read_wrfout_info(INfname,OUT_folder)

   # Variables for saving outputs
   OUT_folder = '/'.join([OUT_folder,DOMAIN,date.strftime('%Y/%m/%d')])
   com = f'mkdir -p {OUT_folder}'
   LG.warning(com)
   os.system(com)

   HH = date.strftime('%H%M')
   date_label = 'valid: ' + date.strftime( fmt ) + 'z\n'
   date_label +=  'GFS: ' + gfs_batch.strftime( fmt ) + '\n'
   date_label += 'plot: ' + creation_date.strftime( fmt+' ' )

   ## READ ALL VARIABLES ########################################################
   bounds, lats,lons,wspd10,wdir10,ua,va,wa, heights, terrain, bldepth,\
   hfx,qcloud,pressure,tc,td,t2m,p,pb,qvapor,MCAPE,rain,blcloudpct,tdif,\
   low_cloudfrac,mid_cloudfrac,high_cloudfrac = wrf_calcs.extract.all_properties(ncfile)

   # useful to setup the extent of the maps
   left   = bounds.bottom_left.lon
   right  = bounds.top_right.lon
   bottom = bounds.bottom_left.lat
   top    = bounds.top_right.lat
   # left   = np.min(wrf.to_np(lons))
   # right  = np.max(wrf.to_np(lons))
   # bottom = np.min(wrf.to_np(lats))
   # top    = np.max(wrf.to_np(lats))

   ## Derived Quantities
   ua10 = -wspd10 * np.sin(np.radians(wdir10))
   va10 = -wspd10 * np.cos(np.radians(wdir10))
   LG.debug(f'ua10: {ua10.shape}')
   LG.debug(f'va10: {va10.shape}')
   wblmaxmin, wstar, blcwbase, hcrit, zsfclcl, zblcl, hglider,\
   ublavgwind, vblavgwind, utop,vtop = wrf_calcs.drjack.calculations(ncfile,wa,heights,\
                                                            terrain,pressure,\
                                                            p,pb,bldepth,\
                                                            hfx,qvapor,qcloud,\
                                                            tc,td)
   blwind = np.sqrt(ublavgwind*ublavgwind + vblavgwind*vblavgwind)
   bltopwind = np.sqrt(utop*utop + vtop*vtop)
   LG.debug(f'BLwind: {blwind.shape}')
   LG.debug(f'BLtopwind: {bltopwind.shape}')
   LG.info('WRF data read')

   ##############################################################################
   #                                    Plots                                   #
   ##############################################################################
   LG.info('Start Plots')
   ## Soundings #################################################################
   f_cities = f'{here}/soundings_{DOMAIN}.csv'
   try:
      Yt,Xt = np.loadtxt(f_cities,usecols=(0,1),delimiter=',',unpack=True)
      names = np.loadtxt(f_cities,usecols=(2,),delimiter=',',dtype=str)
      soundings = [(n,(la,lo))for n,la,lo in zip(names,Yt,Xt)]
      for place,point in soundings:
         lat,lon = point
         if not (left<lon<right and bottom<lat<top): continue
         name = f'{OUT_folder}/{HH}_sounding_{place}.png'
         title = f"{place.capitalize()}"
         title += f" {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
         LG.info(f'Sounding {place}')
         pV,tcV,tdcV,t0V,dateV,uV,vV,latlonV = wrf_calcs.util.sounding(lat,lon,lats,lons,date,ncfile,pressure,tc,td,t2m,ua,va)
         plots.sounding.skewt_plot(pV,tcV,tdcV,t0V,dateV,uV,vV,fout=name,latlon=latlonV,title=title)
   except ValueError: pass


   ## Scalar properties #########################################################
   # Background plots ###########################################################
   dpi = 150
   ## Terrain 
   fname = f'{OUT_folder}/terrain.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting terrain')
      fig,ax,orto = plots.geo.terrain_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted terrain')

   ## Parallel and meridian
   fname = f'{OUT_folder}/meridian.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting meridians')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.parallel_and_meridian(fig,ax,orto,left,right,bottom,top)
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted meridians')
   
   ## Rivers
   fname = f'{OUT_folder}/rivers.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting rivers')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.rivers_plot(fig,ax,orto)
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted rivers')
   
   ## CCAA
   fname = f'{OUT_folder}/ccaa.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting ccaa')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.ccaa_plot(fig,ax,orto)
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted ccaa')
   
   ## Cities
   fname = f'{OUT_folder}/cities.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting cities')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.csv_plot(fig,ax,orto,f'{here}/cities.csv')
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted cities')
   
   ## Citiy Names
   fname = f'{OUT_folder}/cities_names.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting cities names')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.csv_names_plot(fig,ax,orto,f'{here}/cities.csv')
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted cities names')
   
   ## Takeoffs 
   fname = f'{OUT_folder}/takeoffs.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting takeoffs')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.csv_plot(fig,ax,orto,f'{here}/takeoffs.csv')
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted takeoffs')
   
   ## Takeoffs Names
   fname = f'{OUT_folder}/takeoffs_names.png'
   if os.path.isfile(fname):
      LG.info(f'{fname} already present')
   else:
      LG.debug('plotting takeoffs names')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      plots.geo.csv_names_plot(fig,ax,orto,f'{here}/takeoffs.csv')
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info('plotted takeoffs names')
   
   
   # Properties #################################################################
   wrf_properties = {'sfcwind':wspd10, 'blwind':blwind, 'bltopwind':bltopwind,
                     'hglider':hglider, 'wstar':wstar, 'zsfclcl':zsfclcl,
                     'zblcl':zblcl, 'cape':MCAPE, 'wblmaxmin':wblmaxmin,
                     'bldepth': bldepth,  #'bsratio':bsratio,
                     'rain':rain, 'blcloudpct':blcloudpct, 'tdif':tdif,
                     'lowfrac':low_cloudfrac, 'midfrac':mid_cloudfrac,
                     'highfrac':high_cloudfrac}
   
   titles = {'sfcwind':'Viento Superficie', 'blwind':'Viento Promedio',
             'bltopwind':'Viento Altura', 'hglider':'Techo (azul)',
             'wstar':'Térmica', 'zsfclcl':'Base nube', 'zblcl':'Cielo cubierto',
             'cape':'CAPE', 'wblmaxmin':'Convergencias',
             'bldepth': 'Altura Capa Convectiva', 'bsratio': 'B/S ratio',
             'rain': 'Lluvia', 'blcloudpct':'Nubosidad (%)',
             'tdif': 'Prob. Térmica', 'lowfrac':'Nubosidad baja (%)',
             'midfrac': 'Nubosidad media (%)', 'highfrac': 'Nubosidad alta (%)'}
   
   # plot scalars ###############################################################
   ftitles = open(f'{OUT_folder}/titles.txt','w')
   props = ['sfcwind', 'blwind', 'bltopwind', 'wblmaxmin', 'hglider', 'wstar',
            'bldepth', 'cape', 'zsfclcl', 'zblcl', 'tdif', 'rain',
            'blcloudpct', 'lowfrac', 'midfrac', 'highfrac']
   for prop in props:
      LG.debug(f'plotting {prop}')
      factor,vmin,vmax,delta,levels,cmap,units = wrf_calcs.post_process.scalar_props('plots.ini', prop)
      # try: cmap = colormaps[cmap]
      # except: pass  # XXX cmap is already a cmap name
      title = titles[prop]
      title = f"{title} {(date+UTCshift).strftime('%d/%m/%Y-%H:%M')}"
      M = wrf_properties[prop]
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      C = plots.geo.scalar_plot(fig,ax,orto, lons,lats,wrf_properties[prop]*factor,
                         delta,vmin,vmax,cmap, levels=levels,
                         inset_label=date_label)
      fname = f'{OUT_folder}/{HH}_{prop}.png'
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info(f'plotted {prop}')
   
      fname = f'{OUT_folder}/{prop}'
      if os.path.isfile(fname):
         LG.info(f'{fname} already present')
      else:
         LG.debug('plotting colorbar')
         ftitles.write(f"{fname} ; {title}\n")
         plots.geo.plot_colorbar(cmap,delta,vmin,vmax, levels, name=fname,units=units,
                                fs=15,norm=None,extend='max')
         LG.info('plotted colorbar')
   ftitles.close()
   
   ## Vector properties #########################################################
   names = ['sfcwind','blwind','bltopwind']
   winds = [[ua10.values, va10.values],
            [ublavgwind, vblavgwind],
            [utop, vtop]]
   
   for wind,name in zip(winds,names):
      LG.debug(f'Plotting vector {name}')
      fig,ax,orto = plots.geo.setup_plot(reflat,reflon,left,right,bottom,top)
      U = wind[0]
      V = wind[1]
      plots.geo.vector_plot(fig,ax,orto,lons.values,lats.values,U,V, dens=1.5,color=(0,0,0))
      # fname = OUT_folder +'/'+ prefix + name + '_vec.png'
      fname = f'{OUT_folder}/{HH}_{name}_vec.png'
      plots.geo.save_figure(fig,fname,dpi=dpi)
      LG.info(f'Plotted vector {name}')
   
   #XXX shouldn't do this here
   wrfout_folder += '/processed'   #gfs_batch.strftime('/%Y/%m/%d/%H')
   com = f'mkdir -p {wrfout_folder}'
   LG.warning(com)
   os.system(com)
   com = f'mv {INfname} {wrfout_folder}'
   LG.info(com)
   os.system(com)


if __name__ == '__main__':

   import sys
   try: INfname = sys.argv[1]
   except IndexError:
      print('File not specified')
      exit()
   
   DOMAIN = wrf_calcs.extract.get_domain(INfname)

   ################################# LOGGING ####################################
   import logging
   import log_help
   log_file = here+'/'+'.'.join( __file__.split('/')[-1].split('.')[:-1] ) 
   log_file = log_file + f'_{DOMAIN}.log'
   lv = logging.DEBUG
   logging.basicConfig(level=lv,
                    format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
                    datefmt='%Y/%m/%d-%H:%M:%S',
                    filename = log_file, filemode='a')
   LG = logging.getLogger('main')
   if not is_cron: log_help.screen_handler(LG, lv=lv)
   LG.info(f'Starting: {__file__}')
   ##############################################################################

   ## Output folder
   OUT_folder = get_config('plots.ini')

   post_process_file(INfname, OUT_folder)
