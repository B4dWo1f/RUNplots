#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
import log_help
import logging
LG = logging.getLogger(__name__)

import wrf
import numpy as np
import datetime as dt
from netCDF4 import Dataset
import metpy.calc as mpcalc
from metpy.units import units
import util as ut

## True unless RUN_BY_CRON is not defined
is_cron = bool( os.getenv('RUN_BY_CRON') )
import matplotlib as mpl
if is_cron:
   LG.info('Run from cron. Using Agg backend')
   mpl.use('Agg')
import matplotlib.pyplot as plt
try: plt.style.use('mystyle')
except: pass
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from colormaps import WindSpeed

# Get UTCshift automatically
UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))


def get_data_hour(fname,lat,lon):
   ncfile = Dataset(fname)
   # Lats, Lons
   lats = wrf.getvar(ncfile, "lat")
   lons = wrf.getvar(ncfile, "lon")
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = wrf.getvar(ncfile, "pressure")
   # Perturbation pressure_______________________________________[Pa] (nz,ny,nx)
   p = wrf.getvar(ncfile, "P")
   # Base state pressure_________________________________________[Pa] (nz,ny,nx)
   pb = wrf.getvar(ncfile, "PB")
   # Water vapor mixing ratio______________________________________[] (nz,ny,nx)
   qvapor = wrf.getvar(ncfile, "QVAPOR")
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   heights = wrf.getvar(ncfile, "height", units='m') # = z
   # Topography of the terrain ______________________________________[m] (ny,nx)
   terrain = wrf.getvar(ncfile, "ter", units='m') # = HGT
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = wrf.getvar(ncfile, "ua")  # U wind component
   va = wrf.getvar(ncfile, "va")  # V wind component
   wa = wrf.getvar(ncfile, "wa")  # W wind component
   # Planetary Boundary Layer Height_________________________________[m] (ny,nx)
   # Atmospheric Boundary layer thickness above ground
   bldepth = wrf.getvar(ncfile, "PBLH")
   # Surface sensible heat flux in________________________________[W/m²] (ny,nx)
   hfx = wrf.getvar(ncfile, "HFX") 
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = wrf.getvar(ncfile, "tc")
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = wrf.getvar(ncfile, "td", units='degC')
   # Thermal Updraft Velocity (W*)_________________________________[m/s] (ny,nx)
   wstar = ut.calc_wstar( hfx, bldepth )
   # Height of Critical Updraft Strength (hcrit)_____________________[m] (ny,nx)
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth)
   # Height of SFC.LCL_______________________________________________[m] (ny,nx)
   # Cu Cloudbase ~I~where Cu Potential > 0~P~
   zsfclcl = ut.calc_sfclclheight( pressure, tc, td, heights, terrain, bldepth )
   # OvercastDevelopment Cloudbase_______________________________[m?] (nz,ny,nx)
   pmb = 0.01*(p.values+pb.values) # press is vertical coordinate in mb
   zblcl = ut.calc_blclheight(qvapor,heights,terrain,bldepth,pmb,tc)
   # Thermalling Height______________________________________________[m] (ny,nx)
   hglider = np.minimum(np.minimum(hcrit,zsfclcl), zblcl)
   # Point
   i,j = wrf.ll_to_xy(ncfile, lat, lon)  # returns w-e, n-s
   pblh  = bldepth[j,i].values
   hs = np.reshape(heights[:,j,i].values, (-1,1))
   u  = np.reshape(ua[:,j,i].values, (-1,1))
   v  = np.reshape(va[:,j,i].values, (-1,1))
   hcrit = hcrit[j,i]
   wstar = wstar[j,i]
   gnd = terrain[j,i].values
   return hs, u, v, pblh, hcrit, wstar,gnd, lats[j,i].values, lons[j,i].values

def duplicate_first_row(M, value=None):
   """
   This function duplicates the first row like this:
   1 1 1       1 1 1
   2 2 2 ----> 2 2 2
   3 3 3       3 3 3
               3 3 3
   """
   first_row = M[0,:]
   if value != None: first_row = first_row*0+value
   return np.vstack([M,first_row])


def get_meteogram(date0, lat0,lon0, data_fol, OUT_fol,place='', dom='d02',
                                                                fout=None):
   ## Read data
   hours = list(range(8,22))
   fmt_wrfout = '%Y-%m-%d_%H'
   files = []
   for h in hours:
      if h == hours[-1]: h = hours[-2]   #XXX  workaround
      date1 = date0.replace(hour=h) - UTCshift
      files.append(f'{data_fol}/wrfout_{dom}_{date1.strftime(fmt_wrfout)}:00:00')
   if not all([os.path.isfile(x) for x in files]):
      print('Missing files!!!')
      exit()
   heights, windU, windV, BL, Hcrit, Wstar = [], [], [], [], [], []
   for fname in files:
      hs, u, v, pblh, hcrit,wstar,GND,lat,lon = get_data_hour(fname, lat0, lon0)
      heights.append(hs)
      windU.append(u)
      windV.append(v)
      BL.append(pblh)
      Hcrit.append(hcrit)
      Wstar.append(wstar)
   U = np.hstack(windU) * 3.6
   V = np.hstack(windV) * 3.6
   heights = np.hstack(heights)
   Wstar = np.hstack(Wstar)

   ## Cur upper layers ##
   Nup = 10
   U = U[:-Nup,:]
   V = V[:-Nup,:]
   heights = heights[:-Nup,:]
   ######################

   ## Duplicate first row ##
   U = duplicate_first_row(U)
   V = duplicate_first_row(V)
   heights = duplicate_first_row(heights,value=GND)#-10)
   # Derived
   S = np.sqrt(U*U + V*V)
   X = np.array([hours for _ in range(U.shape[0])])
   #########################

   ## Plot
   fig = plt.figure()
   gs = gridspec.GridSpec(2, 1, height_ratios=[45,1])
   fig.subplots_adjust() #wspace=0.1,hspace=0.1)
   ax = plt.subplot(gs[0,0])   # meteogram
   ax1 = plt.subplot(gs[1,0])  # colorbar
   ax.set_yscale('log')

   ## Plot Background Wind Speeds
   C = ax.contourf(X, heights, S, levels=range(0,60,4),
                                  vmin=0, vmax=60, cmap=WindSpeed,
                                  extend='max',zorder=9)
   # rect = Rectangle((-1,-1),24,1e4,facecolor='white',zorder=10,alpha=0.5)
   # ax.add_patch(rect)

   ## Colorbar
   cbar = fig.colorbar(C, cax=ax1, orientation="horizontal")

   ## Plot BL and Thermals
   thermal_color = np.array([255,127,0])/255 # (0.96862745,0.50980392,0.23921569)
   BL_color      = np.array([255,205,142])/255 # (0.90196078,1., 0.50196078)
   # thermal_color = (0.96862745,0.50980392,0.23921569)
   # BL_color      = (0.90196078,1.,        0.50196078)
   W = 0.5
   ax.bar(hours,BL+GND,width=W, color=BL_color, ec=thermal_color,zorder=20)
   ax.bar(hours,Hcrit-GND, width=W-0.15, bottom=GND,
                           color=thermal_color,zorder=21)
   ## Plot Wind barbs
   ax.barbs(X,heights,U,V,length=6,zorder=30)
   ## Plot Terrain Ground
   terrain_color = (0.78235294, 0.37058824, 0.11568627)
   rect = Rectangle((0,0), 24, GND, facecolor=terrain_color, zorder=29)
   ax.add_patch(rect)
   ax.text(0, 0, f'GND: {GND:.0f}', va='bottom', zorder=100,
                                       transform=ax.transAxes)

   ## Title
   ax.set_title(f'{lat:.3f},{lon:.3f}  -  {date0.date()}')

   ## Axis setup
   # X lim
   ax.set_xlim(hours[1]-0.5, hours[-2]+0.5)
   ax.set_xticks(hours[1:-1])
   ax.set_xticklabels([f'{x}:00' for x in hours[1:-1]])
   # Y lim
   ymin = GND-75
   ymax = np.max(BL)+GND+200
   ax.set_ylim([ymin, ymax])
   ax.yaxis.set_major_locator(MultipleLocator(500))
   ax.yaxis.set_minor_locator(MultipleLocator(100))
   ax.yaxis.set_major_formatter(ScalarFormatter())
   ax.yaxis.set_minor_formatter(ScalarFormatter())
   # Grid
   ax.grid(False)

   fig.tight_layout()
   fname = 'adasdfa.png'
   fig.savefig(fname)
   return fname


if __name__ == '__main__':
   date_req = dt.datetime(2021,5,17,12)
   lat,lon = 41.078854,-3.707029 # arcones ladera
   lat,lon = 41.105178018195375, -3.712531733865551     # arcibes cantera
   # lat,lon = 41.078887241417604, -3.7054138385286515  # arcones despegue
   lat,lon = 41.172417,-3.617646 # somo
   lat,lon = 41.131805855213194, -3.684117033819662 # pradena
   lat,lon = 41.16434547255803, -3.571952688735032  # puerto cebollera
   data_folder = '../../Documents/storage/WRFOUT/Spain6_1/'
   OUT_folder = '../../Documents/storage/PLOTS/Spain6_1/'
   place = ''
   fout = 'asfcg.png'
   dom = 'd02'
   fname = get_meteogram(date_req, lat, lon, data_folder, OUT_folder, place, dom, fout)
   print('Saved in:',fname)
