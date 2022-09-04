#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import sys
import datetime as dt
import common

try: day_shift = int(sys.argv[1])
except:
   print('No day provided')
   exit()

today = dt.datetime.now().date()
target_date = today + dt.timedelta(days=day_shift)

output_folder,plots_folder,data_folder = common.get_folders()

import wrf
import numpy as np
from netCDF4 import Dataset
from wrf import getvar, ALL_TIMES
import wrf_calcs.util as ut
import metpy.calc as mcalc
import os
from time import time
import datetime as dt
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
# plt.style.use('meteogram')
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator,ScalarFormatter

import plots.colormaps as mcmaps
################################# LOGGING ####################################
import log_help
import logging
log_file = '.'.join( __file__.split('/')[-1].split('.')[:-1] ) + '.log'
lv = logging.DEBUG
fmt='%(asctime)s:%(name)s:%(levelname)s: %(message)s'
logging.basicConfig(level=lv, format=fmt, datefmt='%Y/%m/%d-%H:%M:%S',
                              filename = log_file, filemode='w')
LG = logging.getLogger('main')
########### Screen Logger (optional) ##########
#sh = logging.StreamHandler()                 #
#sh.setLevel(logging.INFO)                    #
#fmt = '%(asctime)s:%(levelname)s: %(message)s'  #
#fmt = logging.Formatter(fmt)                 #
#sh.setFormatter(fmt)                         #
#LG.addHandler(sh)                            #
##############################################################################
LG.info(f'Starting: {__file__}')



UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))

def get_datetime(fname):
   """
   returns the domain and date from a wrfout file name:
   wrfout_d02_2022-08-02_07:00:00
   """
   fname = fname.split('/')[-1]
   _, dom, d,t = fname.split('_')
   datetime = '_'.join([d,t])
   date = dt.datetime.strptime(datetime,'%Y-%m-%d_%H:%M:%S')
   return dom,date + UTCshift

@log_help.timer(LG)
def get_var(files, prop, units='',cache=None):
   if len(units) >0:
      return getvar(files, prop, timeidx=ALL_TIMES, method="cat", units=units, cache=cache)
   else:
      return getvar(files, prop, timeidx=ALL_TIMES, method="cat", cache=cache)

def duplicate_first_row(M,side='b', value=None):
   """
   This function duplicates the first row like this:
   1 1 1       1 1 1
   2 2 2 ----> 2 2 2
   3 3 3       3 3 3
               3 3 3
   """
   if side == 't':  #XXX check top
      first_row = M[0,:]
      if value != None: first_row = first_row*0.+value
      return np.vstack([M,first_row])
   elif side == 'b':
      first_row = M[-1,:]
      if value != None: first_row = first_row*0+value
      return np.vstack([first_row,M])
   elif side == 'r':
      first_row = np.expand_dims(M[:,-1],axis=1)
      if value != None: first_row = first_row*0+value
      return np.hstack([M,first_row])
   elif side == 'l':
      first_row = np.expand_dims(M[:,0],axis=1)
      if value != None: first_row = first_row*0+value
      return np.hstack([first_row,M])

def plot_map(M):
   fig, ax = plt.subplots()
   ax.contourf(M)
   ax.set_title(f'{target_date} {hours_row[itime]}:00')
   fig.tight_layout()
   plt.show()


def expand_sides(M):
   M = duplicate_first_row(M, side='b')
   M = duplicate_first_row(M, side='l')
   M = duplicate_first_row(M, side='r')
   return M

"""
The data will be extracted and processed into a matrix with the times in one axis
and the heights in the other:

        /  9  10  11  12  13 ... \
        |  9  10  11  12  13 ... |
        |  9  10  11  12  13 ... |
 hours: |  9  10  11  12  13 ... |
        |  9  10  11  12  13 ... |
        |  9  10  11  12  13 ... |
        |  9  10  11  12  13 ... |
        \  9  10  11  12  13 ... /


         / 40 .... \
         | 35 .... |
         | 30 .... |
 Wspeed: | 23 .... |
         | 25 .... |
         | 23 .... |
         \ 19 .... /
"""

##XXX  Inputs
#target_date = dt.date(2022,8,23)
#itime = 9
#folder = '../../Documents/storage/WRFOUT/Spain6_1'
#############

com = f"ls {output_folder}/processed/wrfout_d02_{target_date.strftime('%Y-%m-%d')}_*"
files = os.popen(com).read().strip().split()
files = sorted(files)
dates = [get_datetime(f)[1] for f in files]
hours_row = [d.time().hour for d in dates]



fname = 'soundings_d02.csv'
Points = np.loadtxt(fname,usecols=(0,1),delimiter=',')
names = np.loadtxt(fname,usecols=(2,),delimiter=',',dtype=str)
titles = np.loadtxt(fname,usecols=(3,),delimiter=',',dtype=str)


import wrf_calcs.extract as ex

#LG.info('Reading')
## Creating a simple test list with three timesteps
#wrflist = [Dataset(x) for x in files]
#terrain   = get_var(wrflist, "ter",        units='m')      # (nt,ny,nx)
#heights   = get_var(wrflist, "height",     units='m')      # (nt,ny,nx)
#pressure  = get_var(wrflist, "p",          units='hPa')    # (nt,nz,ny,nx) 
#PB        = get_var(wrflist, "PB")   #!!!  Pa  !!!         # (nt,nz,ny,nx) 
#T         = get_var(wrflist, "temp",       units='degC')   # (nt,nz,ny,nx)
#TD        = get_var(wrflist, "td",         units='degC')   # (nt,nz,ny,nx)
#T2m       = get_var(wrflist, "T2")   #!!!  Kelvin  !!!     # (nt,ny,nx)
#TD2m      = get_var(wrflist, "td2",        units='degC')   # (nt,ny,nx)
#HFX       = get_var(wrflist, "HFX")  #!!!  W m-2   !!!     # (nt,ny,nx)
#low, mid, high= get_var(wrflist, "cloudfrac")              # (nt,ny,nx)
#umet,vmet = get_var(wrflist, "uvmet",      units='km h-1') # (nt,nz,ny,nx)
#wspd      = get_var(wrflist, "uvmet_wspd", units='km h-1') # (nt,nz,ny,nx)
#umet10,vmet10 = get_var(wrflist, "uvmet10",units='km h-1') # (nt,ny,nx)
#BL        = get_var(wrflist, "PBLH") # meters AGL          # (nt,ny,nx)
## BL += terrain                        # meters above MSL    # (nt,ny,nx)
#rain      = get_var(wrflist, "RAINC")
#rain     += get_var(wrflist, "RAINNC")
#rain     += get_var(wrflist, "RAINSH")
#qvapor    = get_var(wrflist, "QVAPOR")


told = time()
# Creating a simple test list with three timesteps
wrflist = [Dataset(x) for x in files]
mycache = ex.get_cache(wrflist)
terrain   = get_var(wrflist, "ter",        units='m', cache=mycache)      # (nt,ny,nx)
heights   = get_var(wrflist, "height",     units='m', cache=mycache)      # (nt,ny,nx)
pressure  = get_var(wrflist, "p",          units='hPa', cache=mycache)    # (nt,nz,ny,nx) 
PB        = get_var(wrflist, "PB", cache=mycache)   #!!!  Pa  !!!         # (nt,nz,ny,nx) 
T         = get_var(wrflist, "temp",       units='degC', cache=mycache)   # (nt,nz,ny,nx)
TD        = get_var(wrflist, "td",         units='degC', cache=mycache)   # (nt,nz,ny,nx)
T2m       = get_var(wrflist, "T2")   #!!!  Kelvin  !!!     # (nt,ny,nx)
TD2m      = get_var(wrflist, "td2",        units='degC', cache=mycache)   # (nt,ny,nx)
HFX       = get_var(wrflist, "HFX", cache=mycache)  #!!!  W m-2   !!!     # (nt,ny,nx)
low, mid, high= get_var(wrflist, "cloudfrac", cache=mycache)              # (nt,ny,nx)
umet,vmet = get_var(wrflist, "uvmet",      units='km h-1', cache=mycache) # (nt,nz,ny,nx)
wspd      = get_var(wrflist, "uvmet_wspd", units='km h-1', cache=mycache) # (nt,nz,ny,nx)
umet10,vmet10 = get_var(wrflist, "uvmet10",units='km h-1', cache=mycache) # (nt,ny,nx)
BL        = get_var(wrflist, "PBLH") # meters AGL          # (nt,ny,nx)
# BL += terrain                        # meters above MSL    # (nt,ny,nx)
rain      = get_var(wrflist, "RAINC", cache=mycache)
rain     += get_var(wrflist, "RAINNC", cache=mycache)
rain     += get_var(wrflist, "RAINSH", cache=mycache)
qvapor    = get_var(wrflist, "QVAPOR", cache=mycache)
## Dr Jack ######################################################################
pmb = .01*(pressure.values + PB.values)  #XXX WTF???
wstar = ut.calc_wstar(HFX.values, BL.values)
hcrit = ut.calc_hcrit(wstar, terrain.values, BL.values)
cumulus  = ut.calc_sfclclheight(pressure, T, TD, heights, terrain, BL)
overcast = ut.calc_blclheight(qvapor.values,heights.values,terrain.values,
                              BL.values,pmb,T.values)
#################################################################################
hours = np.stack([hours_row for _ in range(wspd.shape[1])])
LG.debug(f'Time reading: {time()-told}')

LG.debug(f'hfx: {HFX.values.shape}')
LG.debug(f'BL: {BL.values.shape}')
LG.debug(f'heights: {heights.values.shape}')
LG.debug(f'P: {pressure.values.shape}')
LG.debug(f'wspd: {wspd.values.shape}')
LG.debug(f'U: {umet.values.shape}')
LG.debug(f'V: {vmet.values.shape}')
LG.debug(f'BL: {BL.values.shape}')
LG.debug(f'T2: {T2m.values.shape}')
LG.debug(f'TD2: {TD2m.values.shape}')
LG.debug(f'HFX: {HFX.values.shape}')
LG.debug(f'wstar: {wstar.shape}')
LG.debug(f'hcrit: {hcrit.shape}')
LG.debug(f'terrain: {terrain.values.shape}')
LG.debug(f'low clouds: {low.values.shape}')
LG.debug(f'mid clouds: {mid.values.shape}')
LG.debug(f'high clouds: {high.values.shape}')


BL_color = np.array([255,205,142])/255
thermal_color = np.array([255,127,0])/255 # (0.96862745,0.50980392,0.23921569)
# terrain_color = (0.78235294, 0.37058824, 0.11568627)
terrain_color = np.array([158,65,12])/255
rain_color = np.array([154,224,228])/255
bbox_barbs = dict(spacing=.2,emptybarb=0.2, width=.5, height=.5)



for iplace in range(len(Points)):
   told = time()
   # Select index
   LG.info(f'Doing: {names[iplace]} {Points[iplace]}')
   lat,lon = Points[iplace]
   j,i = wrf.ll_to_xy(wrflist[0], lat, lon)

   # Prepare properties
   H = heights[:,:,i,j].transpose()
   W = wspd[:,:,i,j].transpose()
   GND = terrain[0,i,j].values
   # Cloud percentage
   #XXX fix order!!!! WTF??!!!
   img_cloud_pct = np.stack([low[:,i,j],
                             mid[:,i,j],
                             high[:,i,j]])
   img_cloud_pct = np.flipud(img_cloud_pct)
   ###########################
   hours_cloud = np.stack([[x-.5 for x in hours_row] for _ in range(img_cloud_pct.shape[0])])
   aux = []
   for ic in range(img_cloud_pct.shape[0]):
      aux.append([ic for _ in range(len(hours_row))])
   Ycloud = np.flipud(np.array(aux))

   # Plotting
   # fig, ax = plt.subplots()
   hr = [2,17,1]
   l = .08
   r = .98
   t = .97
   gs_plots = plt.GridSpec(3, 1, height_ratios=hr,hspace=0.,top=t,
                                                 left=l,right=r,bottom=0.05)
   gs_cbar  = plt.GridSpec(3, 1, height_ratios=hr,hspace=0.5,top=t,
                                                 left=l,right=r, bottom=0.025)
   fig = plt.figure(figsize=(10, 12))
   # fig.subplots_adjust() #hspace=[0,0.2])
   ax =  fig.add_subplot(gs_plots[1,:])   # meteogram
   ax0 = fig.add_subplot(gs_plots[0,:], sharex=ax)  # clouds
   ax1 = fig.add_subplot(gs_cbar[2,:])  # colorbar

   # Clouds
   ax0.imshow(img_cloud_pct, extent=[min(hours_row),max(hours_row),0,2],
              cmap='Greys',vmin=0,vmax=1, aspect='auto')
   ax0.set_yticks([1/3,1,5/3])
   ax0.set_yticklabels(['low','mid','high'])
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.set_title(f'{titles[iplace].capitalize()} {target_date}')

   ## Meteogram
   # Wind speed background
   # Duplicate bottom
   hourss = duplicate_first_row(hours, side='b')
   W = duplicate_first_row(W, side='b')
   H = duplicate_first_row(H, side='b', value=GND-100)
   # Duplicate left
   hourss = duplicate_first_row(hourss, side='l', value=min(hours_row)-1)
   W = duplicate_first_row(W, side='l')
   H = duplicate_first_row(H, side='l')
   # Duplicate right
   hourss = duplicate_first_row(hourss, side='r', value=max(hours_row)+1)
   W = duplicate_first_row(W, side='r')
   H = duplicate_first_row(H, side='r')
   C = ax.contourf(hourss,H,W, levels=range(0,60,4), vmin=0, vmax=60, cmap=mcmaps.WindSpeed, extend='max',zorder=0,alpha=.7)
   # Thermals
   # ax.bar(hours_row, BL[:,i,j], color=BL_color, ec=thermal_color, zorder=1)
   ax.bar(hours_row, hcrit[:,i,j], width=0.6, color=thermal_color, zorder=2)
   for x,y,w in zip(hours_row, hcrit[:,i,j], wstar[:,i,j]):
      if w == 0.: continue
      if y-GND < 100 : continue
      t = f'{w}'[:3]+' m/s'
      y -= 20  # to align the box
      ax.text(x-.25,y,t, ha='right',va='top',rotation='vertical',backgroundcolor=(1,1,1,.5),zorder=2)
   # Clouds
   ax.bar(hours_row, 9000, bottom=overcast[:,i,j], width=1, color=(.3,.3,.3,.7), zorder=4)
   ax.bar(hours_row, 9000, bottom=cumulus[:,i,j], width=.7, hatch='O', color=(.3,.2,.2,.7), zorder=4)
   # Windbarbs
   hourss = duplicate_first_row(hours)
   ax.barbs(hours, heights[:,:,i,j].transpose(),
            umet[:,:,i,j].transpose(), vmet[:,:,i,j].transpose(),
            length=5,sizes=bbox_barbs, zorder=3)
   ax.barbs(hours_row, terrain[:,i,j], umet10[:,i,j], vmet10[:,i,j],
            length=6,sizes=bbox_barbs, color='r', lw=3,zorder=100)
   # Rain
   rain_dif = np.diff(rain[:,i,j])
   rain_dif = np.insert(rain_dif,0,rain_dif[0])
   ax.bar(hours_row, GND+rain_dif*100,
          color=rain_color, width=.3, zorder=99)
   # Ground
   rect = Rectangle((-24,0), 48, GND, facecolor=terrain_color, zorder=99)
   ax.add_patch(rect)
   # Settings
   ax.set_xticks(hours_row)
   ax.set_xticklabels([f'{x}:00' for x in hours_row])
   ax.set_xlim([min(hours_row)-.5,max(hours_row)+.5])
   ymin = GND-100
   ymax = np.max([np.max(BL[:,i,j]),
                  np.max(hcrit[:,i,j]),
                  np.max(cumulus[:,i,j]),
                  np.max(overcast[:,i,j]) ]) +500
   # ax.set_ylim([GND-100, np.max()+500])
   ax.set_ylim([ymin, ymax])
   ax.yaxis.set_major_locator(MultipleLocator(500))
   ax.yaxis.set_minor_locator(MultipleLocator(100))
   ax.set_ylabel('Height (m)')
   # Grid
   ax.grid(False)

   # Colorbar
   cbar = fig.colorbar(C, cax=ax1, orientation="horizontal")
   ax1.set_ylabel('km/h')

   # fig.tight_layout()
   LG.debug(f'Plotting: {time()-told}')
   folder = f"{plots_folder}/d02/{target_date.strftime('%Y/%m/%d')}"
   fig.savefig(f"{folder}/meteogram_{names[iplace]}.png")
   LG.info(f'Saved: {folder}/meteogram_{names[iplace]}.png')
   # plt.show()
