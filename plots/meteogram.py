#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
## True unless RUN_BY_CRON is not defined
is_cron = bool( os.getenv('RUN_BY_CRON') )
import matplotlib as mpl
if is_cron:
   LG.info('Run from cron. Using Agg backend')
   mpl.use('Agg')
import matplotlib.pyplot as plt
try: plt.style.use('mystyle')
except: pass
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator,ScalarFormatter
from . import colormaps as mcmaps

def meteogram(GND,hours,X,heights,BL,Hcrit,Zover,Zcu,S,U,V,PCT_low,PCT_mid,PCT_high,title='',fout='meteogram.png'):
   """
   hours: only plot hours[1:-1]. hours[0] is GFS-smothen. hours[-1] is a
          duplicate of hours[-2]
   X: X grid (vertical repetitions of hours)
   heights: Y grid
   """
   log = True
   ## Plot
   gs_plots = plt.GridSpec(3, 1, height_ratios=[2,17,1],hspace=0.,top=0.95,right=0.95,bottom=0.05)
   gs_cbar  = plt.GridSpec(3, 1, height_ratios=[2,17,1],hspace=0.5,top=0.95,right=0.95, bottom=0)
   fig = plt.figure()
   # fig.subplots_adjust() #hspace=[0,0.2])
   ax =  fig.add_subplot(gs_plots[1,:])   # meteogram
   ax0 = fig.add_subplot(gs_plots[0,:], sharex=ax)  # clouds
   ax1 = fig.add_subplot(gs_cbar[2,:])  # colorbar
   if log: ax.set_yscale('log')

   ## % of low-mid-high-clouds
   img_cloud_pct = np.vstack((PCT_low,PCT_mid,PCT_high))
   Xcloud = np.array([hours for _ in range(img_cloud_pct.shape[0])])
   Ycloud = 0*Xcloud.transpose() + np.array(range(img_cloud_pct.shape[0]))
   Ycloud = Ycloud.transpose()
   print(img_cloud_pct)
   ax0.contourf(Xcloud,Ycloud,img_cloud_pct, origin='lower',
                                             cmap='Greys', vmin=0, vmax=1)
   ax0.set_yticks(range(img_cloud_pct.shape[0]))
   ax0.set_yticklabels(['low','mid','high'])
   ax0.set_ylim(0,2)
   plt.setp(ax0.get_xticklabels(), visible=False)
   ax0.grid(False)
   ## Plot Background Wind Speeds
   C = ax.contourf(X, heights, S, levels=range(0,60,4),
                                  vmin=0, vmax=60, cmap=mcmaps.WindSpeed,
                                  extend='max',zorder=9)
   ## alpha workaround
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
   if len(title) > 0: ax0.set_title(title)

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
