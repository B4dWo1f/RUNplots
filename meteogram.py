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
   my_vars = ("P", "PSFC", "PB", "PH", "PHB","T", "QVAPOR", "HGT", "U", "V","W")
   my_cache = wrf.extract_vars(ncfile, wrf.ALL_TIMES, (my_vars))
   # Lats, Lons
   lats = wrf.getvar(ncfile, "lat",cache=my_cache)
   lons = wrf.getvar(ncfile, "lon",cache=my_cache)
   # Pressure___________________________________________________[hPa] (nz,ny,nx)
   pressure = wrf.getvar(ncfile, "pressure", cache=my_cache)
   # Perturbation pressure_______________________________________[Pa] (nz,ny,nx)
   p = wrf.getvar(ncfile, "P", cache=my_cache)
   # Base state pressure_________________________________________[Pa] (nz,ny,nx)
   pb = wrf.getvar(ncfile, "PB", cache=my_cache)
   # Water vapor mixing ratio______________________________________[] (nz,ny,nx)
   qvapor = wrf.getvar(ncfile, "QVAPOR", cache=my_cache)
   # Planetary Boundary Layer Height_________________________________[m] (ny,nx)
   # Atmospheric Boundary layer thickness above ground
   bldepth = wrf.getvar(ncfile, "PBLH", cache=my_cache)
   # Surface sensible heat flux in________________________________[W/m²] (ny,nx)
   hfx = wrf.getvar(ncfile, "HFX", cache=my_cache) 
   terrain = wrf.getvar(ncfile, "ter", units='m', cache=my_cache)
   terrain1= wrf.getvar(ncfile,'HGT')
   # Vertical levels of the grid__________________________________[m] (nz,ny,nx)
   heights = wrf.getvar(ncfile, "height", units='m', cache=my_cache) # = z
   # Topography of the terrain ______________________________________[m] (ny,nx)
   terrain = wrf.getvar(ncfile, "ter", units='m', cache=my_cache) # = HGT
   # Wind_______________________________________________________[m/s] (nz,ny,nx)
   ua = wrf.getvar(ncfile, "ua", cache=my_cache)  # U wind component
   va = wrf.getvar(ncfile, "va", cache=my_cache)  # V wind component
   wa = wrf.getvar(ncfile, "wa", cache=my_cache)  # W wind component
   # Temperature_________________________________________________[°C] (nz,ny,nx)
   tc = wrf.getvar(ncfile, "tc", cache=my_cache)
   # Temperature Dew Point_______________________________________[°C] (nz,ny,nx)
   td = wrf.getvar(ncfile, "td", units='degC', cache=my_cache)
   # Thermal Updraft Velocity (W*)_________________________________[m/s] (ny,nx)
   wstar = ut.calc_wstar( hfx, bldepth )
   # Height of Critical Updraft Strength (hcrit)_____________________[m] (ny,nx)
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth)
   # Height of SFC.LCL_______________________________________________[m] (ny,nx)
   # Cu Cloudbase ~I~where Cu Potential > 0~P~
   zsfclcl = ut.calc_sfclclheight( pressure, tc, td, heights, terrain, bldepth )
   # OvercastDevelopment Cloudbase__________________________________[m?] (ny,nx)
   pmb = 0.01*(p.values+pb.values) # press is vertical coordinate in mb
   zblcl = ut.calc_blclheight(qvapor,heights,terrain,bldepth,pmb,tc)
   # Mask zsfclcl, zblcl________________________________________________________
   ## Mask Cu Pot > 0
   zsfclcldif = bldepth + terrain - zsfclcl
   null = 0. * zsfclcl
   # cu_base_pote = np.where(zsfclcldif>0, zsfclcl, null)
   zsfclcl = np.where(zsfclcldif>0, zsfclcl, null)
   ## Mask Overcast dev Pot > 0
   zblcldif = bldepth + terrain - zblcl
   null = 0. * zblcl
   # over_base_pote = np.where(zblcldif>0, zblcl, null)
   zblcl = np.where(zblcldif>0, zblcl, null)
   # Clouds__________________________________________________________[%] (ny,nx)
   low_cloudfrac  = wrf.getvar(ncfile, "low_cloudfrac", cache=my_cache)
   mid_cloudfrac  = wrf.getvar(ncfile, "mid_cloudfrac", cache=my_cache)
   high_cloudfrac = wrf.getvar(ncfile, "high_cloudfrac", cache=my_cache)
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
   zblcl = zblcl[j,i]
   zsfclcl = zsfclcl[j,i]
   low_cloudfrac = low_cloudfrac[j,i].values
   mid_cloudfrac = mid_cloudfrac[j,i].values
   high_cloudfrac = high_cloudfrac[j,i].values
   gnd = terrain[j,i].values
   lat = lats[j,i].values
   lon = lons[j,i].values
   return lat,lon, hs, u, v, pblh, hcrit, wstar,gnd, zsfclcl, zblcl,\
          low_cloudfrac,mid_cloudfrac,high_cloudfrac

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
   heights, windU,windV, BL,Hcrit,Wstar,Zcu,Zover = [],[],[],[],[],[],[],[]
   PCT_low,PCT_mid,PCT_high = [],[],[]
   for fname in files:
      lat,lon, hs, u, v, pblh, hcrit, wstar, GND, cumulus, overcast,\
      lowpct,midpct,highpct = get_data_hour(fname, lat0, lon0)
      # hs, u, v, pblh, hcrit,wstar,GND,lat,lon = get_data_hour(fname, lat0, lon0)
      heights.append(hs)
      windU.append(u)
      windV.append(v)
      BL.append(pblh)
      Hcrit.append(hcrit)
      Wstar.append(wstar)
      Zcu.append(cumulus)
      Zover.append(overcast)
      PCT_low.append(lowpct)
      PCT_mid.append(midpct)
      PCT_high.append(highpct)
   U = np.hstack(windU) * 3.6
   V = np.hstack(windV) * 3.6
   heights = np.hstack(heights)
   Wstar = np.hstack(Wstar)
   BL = np.hstack(BL)
   Zcu = np.hstack(Zcu)
   Zover = np.hstack(Zover)
   hours = np.hstack(hours)
   # PCT_low = np.hstack(PCT_low)
   # PCT_mid = np.hstack(PCT_mid)
   # PCT_high = np.hstack(PCT_high)

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
   gs_plots = plt.GridSpec(3, 1, height_ratios=[2,17,1],hspace=0.,top=0.95,right=0.95,bottom=0.05)
   gs_cbar  = plt.GridSpec(3, 1, height_ratios=[2,17,1],hspace=0.5,top=0.95,right=0.95, bottom=0)
   fig = plt.figure()
   # fig.subplots_adjust() #hspace=[0,0.2])
   ax0 = fig.add_subplot(gs_plots[0,:])  # clouds
   ax =  fig.add_subplot(gs_plots[1,:])   # meteogram
   ax1 = fig.add_subplot(gs_cbar[2,:])  # colorbar
   # ax.set_yscale('log')

   ## % of low-mid-high-clouds
   img_cloud_pct = np.vstack((PCT_low,PCT_mid,PCT_high))
   ax0.imshow(img_cloud_pct, origin='lower', cmap='Greys',aspect='auto')
   ax0.set_xticks([])
   ax0.set_yticks([])
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
   W = 0.6
   ax.bar(hours,BL+GND,width=W, color=BL_color, ec=thermal_color,zorder=20)
   ax.bar(hours,Hcrit-GND, width=W-0.15, bottom=GND,
                           color=thermal_color,zorder=21)
   ## Clouds
   ax.bar(hours,Zover+100, bottom=Zover, width=1, color=(.4,.4,.4),
                                                           zorder=21, alpha=0.8)
   # ax.bar(hours,Zcu+100,bottom=Zcu,width=W+0.15, color=(.3,.3,.3), hatch='O', 
   cu_top = np.where(Zcu>0,BL+GND-Zcu,-100)
   ax.bar(hours,cu_top, bottom=Zcu, width=W+.2,color=(.3,.3,.3),hatch='O', 
                                                           zorder=22, alpha=0.8)
   # overcast_top = np.where(BL+GND > Zover,BL+GND-Zover,1000)
   # ax.bar(hours,overcast_top, bottom=Zover, width=0.9, color=(.4,.4,.4),
   #                                                       zorder=21, alpha=0.75)
   # cu_top = np.where(BL+GND > Zcu,BL+GND-Zcu,Zcu+100)
   # ax.bar(hours,cu_top, bottom=Zcu, width=W+0.15, color=(.3,.3,.3), hatch='O', 
   #                                                       zorder=22, alpha=0.75)
   ## Plot Wind barbs
   ax.barbs(X,heights,U,V,length=6,zorder=30)
   ## Plot Terrain Ground
   terrain_color = (0.78235294, 0.37058824, 0.11568627)
   rect = Rectangle((0,0), 24, GND, facecolor=terrain_color, zorder=29)
   ax.add_patch(rect)
   ax.text(0, 0, f'GND: {GND:.0f}', va='bottom', zorder=100,
                                       transform=ax.transAxes)

   ## Title
   ax0.set_title(f'{lat:.3f},{lon:.3f}  -  {date0.date()}')

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
   ax.set_ylabel('Height (m)')
   # Grid
   ax.grid(False)

   # fig.tight_layout()
   fig.savefig(fout,bbox_inches='tight')
   return fout


if __name__ == '__main__':
   date_req = dt.datetime(2021,5,19)
   lat,lon = 41.078854,-3.707029 # arcones ladera
   lat,lon = 41.105178018195375, -3.712531733865551     # arcibes cantera
   # lat,lon = 41.078887241417604, -3.7054138385286515  # arcones despegue
   lat,lon = 41.131805855213194, -3.684117033819662 # pradena
   lat,lon = 41.16434547255803, -3.571952688735032  # puerto cebollera
   lat,lon = 41.172417,-3.617646 # somo
   data_folder = '../../Documents/storage/WRFOUT/Spain6_1'
   OUT_folder = '../../Documents/storage/PLOTS/Spain6_1'
   place = ''
   fout = 'meteogram.png'
   dom = 'd02'
   fname = get_meteogram(date_req, lat, lon, data_folder, OUT_folder, place, dom, fout)
   print('Saved in:',fname)
