#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import log_help
import logging
LG = logging.getLogger(__name__)
import os
here = os.path.dirname(os.path.realpath(__file__))

def recompile(f90):
   f90_root = '.'.join(f90.split('.')[0:-1])
   diff = os.popen(f'cd {here} && diff {f90} .{f90}').read().strip()
   if len(diff) > 0:
      LG.warning(f'Compiling {f90}')
      com = f'cd {here} && f2py3 -c -m {f90_root} {f90} && cp {f90} .{f90} && cd -'
      LG.warning(com)
      os.system(com)
   else: LG.debug('Already compiled')

# Recompile DrJack's Fortran if necessary
recompile(f'drjack_num.f90')
from . import drjack_num
import wrf
import plots
# import sounding_plot as SP
# import plot_functions as PL
import numpy as np

from configparser import ConfigParser, ExtendedInterpolation
from os.path import expanduser

def get_domain(fname):
   return fname.split('/')[-1].replace('wrfout_','').split('_')[0]

def check_directory(path, stop=False):
   if os.path.isdir(path):
      LG.debug(f'{path} exists')
   else:
      if stop:
         LG.critical(f'{path} does not exist')
         exit()
      else:
         LG.warning(f'{path} does not exist')
         com = f'mkdir -p {path}'
         LG.debug(com)
         os.system(com)
         LG.info(f'created {path}')

def get_outfolder(fname):
   """
   Load the config options and return it as a class
   """
   # LG.info(f'Loading config file: {fname}')
   # if not os.path.isfile(fname): return None
   config = ConfigParser(inline_comment_prefixes='#')
   config._interpolation = ExtendedInterpolation()
   config.read(fname)
   wrfout_folder = expanduser(config['run']['output_folder'])
   plots_folder = expanduser(config['run']['plots_folder'])
   return wrfout_folder, plots_folder


def maskPot0(M, terrain,bldepth):
   """
   input: arrays
   """
   Mdif = terrain+bldepth - M
   null = 0. * M
   return np.where( Mdif>0, M, null )



def calc_wblmaxmin(linfo, wa, heights, terrain, bldepth):
   """
   Wrapper for DrJack's transposes
   """
   wblmaxmin = drjack_num.calc_wblmaxmin(linfo, wa.transpose(),
                                            heights.transpose(),
                                            terrain.transpose(),
                                            bldepth.transpose())
   return wblmaxmin.transpose()

def calc_wstar( hfx, bldepth ):
   return drjack_num.calc_wstar( hfx.transpose(), bldepth.transpose() ).transpose()

def calc_blcloudbase( qcloud,  heights, terrain, bldepth, cwbasecriteria,
                                                       maxcwbasem, laglcwbase):
   blcwbase = drjack_num.calc_blcloudbase( qcloud.transpose(),
                                       heights.transpose(),
                                       terrain.transpose(),
                                       bldepth.transpose(),
                                       cwbasecriteria, maxcwbasem, laglcwbase)
   return blcwbase.transpose()

def calc_hcrit( wstar, terrain, bldepth):
   hcrit = drjack_num.calc_hcrit( wstar.transpose(), terrain.transpose(),
                              bldepth.transpose() )
   return hcrit.transpose()

def calc_blclheight(qvapor,heights,terrain,bldepth,pmb,tc):
   qvapor = qvapor.transpose()
   heights = heights.transpose()
   terrain = terrain.transpose()
   bldepth = bldepth.transpose()
   pmb = pmb.transpose()
   tc = tc.transpose()
   qvaporblavg = drjack_num.calc_blavg( qvapor, heights, terrain, bldepth)
   # pmb=var = 0.01*(p.values+pb.values) # press is vertical coordinate in mb
   zblcl = drjack_num.calc_blclheight( pmb, tc, qvaporblavg, heights, terrain,
                                   bldepth )
   return zblcl.transpose()


def calc_sfclclheight( pressure, tc, td, heights, terrain, bldepth):
   # Cu Cloudbase ~I~where Cu Potential > 0~P~
   zsfclcl = drjack_num.calc_sfclclheight( pressure.transpose(),
                                       tc.transpose(), td.transpose(),
                                       heights.transpose(),
                                       terrain.transpose(),
                                       bldepth.transpose() )
   return zsfclcl.transpose()

def calc_blavg(X, heights,terrain,bldepth):
   Xblavgwind = drjack_num.calc_blavg(X.transpose(), heights.transpose(),
                                                 terrain.transpose(),
                                                 bldepth.transpose())
   return Xblavgwind.transpose()

def calc_bltopwind(uEW,vNS,heights,terrain,bldepth):
   utop,vtop = drjack_num.calc_bltopwind(uEW.transpose(),
                                     vNS.transpose(),
                                     heights.transpose(),
                                     terrain.transpose(),
                                     bldepth.transpose())
   return utop.transpose(), vtop.transpose()



def sounding(lat,lon,lats,lons,date,ncfile,pressure,tc,td,t0,ua,va):
   """
   lat,lon: spatial coordinates for the sounding
   date: UTC date-time for the sounding
   ncfile: ntcd4 Dataset from the WRF output
   tc: Model temperature in celsius
   tdc: Model dew temperature in celsius
   t0: Model temperature 2m above ground
   ua: Model X wind (m/s)
   va: Model Y wind (m/s)
   fout: save fig name
   """
   LG.info('Starting sounding')
   i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
   # Get sounding data for specific location
   # h = heights[:,i,j]
   latlon = f'({lats[j,i].values:.3f},{lons[j,i].values:.3f})'
   nk,nj,ni = pressure.shape
   p = pressure[:,j,i]
   tc = tc[:,j,i]
   tdc = td[:,j,i]
   u = ua[:,j,i]
   v = va[:,j,i]
   t0 = t0[j,i]
   LG.info('calling skewt plot')
   return p,tc,tdc,t0,date,u,v,latlon


def cross_path(start,end):
# start = 41.260096661378306, -3.6981749563104716
# end = 40.78691893349439, -3.6903966736445306
# cross_path(start,end)
   start = wrf.ll_to_xy(ncfile, start[0], start[1])
   end = wrf.ll_to_xy(ncfile, end[0], end[1])

   start_point = wrf.CoordPair(x=start[0], y=start[1])
   end_point   = wrf.CoordPair(x=end[0], y=end[1])

   levels = np.linspace(900,3000,100)
   p_vert = wrf.vertcross(wspd, heights, start_point=start_point,
                                         end_point=end_point,
                                         levels=levels,
                                         latlon=True)
   fig, ax = plt.subplots()
   dx = 0,p_vert.shape[1]
   ax.imshow(p_vert,origin='lower',extent=[*dx,900,3000])
   ax.set_aspect(1/100)
   ax.set_title('Wind speed')
   fig.tight_layout()
   plt.show()


def fermi(x,x0,t=1):
   return 1/(np.exp((x-x0)/(t))+1)

# Obsolete? ####################################################################
def strip_plot(fig,ax,lims,aspect,fname,dpi=65):
   ax.set_aspect(aspect)
   ax.set_xlim(lims[0:2])
   ax.set_ylim(lims[2:4])
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   plt.axis('off')
   fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
   fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0,
                      dpi=dpi, quality=90)   # compression
