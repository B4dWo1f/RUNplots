#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
import log_help
import logging
LG = logging.getLogger(__name__)

## True unless RUN_BY_CRON is not defined
is_cron = bool( os.getenv('RUN_BY_CRON') )
import matplotlib as mpl
if is_cron:
   LG.info('Run from cron. Using Agg backend')
   mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LightSource, BoundaryNorm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from . import colormaps as mcmaps
from scipy.interpolate import interp1d

# Map
import numpy as np
import rasterio
from rasterio.merge import merge
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

# Sounding
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc

import wrf_calcs

p2m = mpcalc.pressure_to_height_std
m2p = mpcalc.height_to_pressure_std

def get_bottom_temp(ax,T,P):
   """
   Returns the temperature in the X axis of any point in the skew-T graph
   """
   # Lower axis
   x0,x1 = ax.get_xlim()
   y0,y1 = ax.get_ylim()
   X0,X1 = ax.upper_xlim
   # Upper axis
   H = (x0-X0)/np.tan(np.radians(ax.rot))
   DX = x0-X0
   hp = H*(P.magnitude-y0)/(y1-y0)
   dx = hp*DX/H
   return T.magnitude+dx


@log_help.timer(LG)
# def skewt_plot(p,tc,tdc,t0,date,u=None,v=None,fout='sounding.png',latlon='',title='',show=False):
def skewt_plot(p,tc,tdc,t0,date,u,v,gnd,cu_base_p,cu_base_m,cu_base_t,Xcloud,Ycloud,cloud,lcl_p,lcl_t,parcel_prof,fout='sounding.png',latlon='',title='',rot=30,show=False):
   """
   Layout             ________________________
                  150|                 |C|Hod||<-- ax_hod
   ax1=skew_top.ax-->|                 |L|___||
                  500|_________________|O|  I |
                  500|                 |U|  N |
                     |                 |D|W T |
                     |                 |S|I E |
                     |    SOUNDING     | |N N |
                     |                 | |D S |
                     |                 | |  I |
                     |                 | |  T |
                 1000|_________________|_|__Y_|
                       ^                ^    ^
                ax0=skew_bot.ax  ax_clouds  ax_wind
   h: heights
   p: pressure
   tc: Temperature [C]
   tdc: Dew point [C]
   date: date of the forecast
   u,v: u,v wind components
   adapted from:
   https://geocat-examples.readthedocs.io/en/latest/gallery/Skew-T/NCL_skewt_3_2.html#sphx-glr-gallery-skew-t-ncl-skewt-3-2-py
   """
   mpl.rcParams['axes.facecolor'] = (1,1,1,1)
   mpl.rcParams['figure.facecolor'] = (1,1,1,1)
   mpl.rcParams["savefig.facecolor"] = (1,1,1,1)
   mpl.rcParams['font.family'] = 'serif'
   mpl.rcParams['font.size'] = 15.0
   mpl.rcParams['mathtext.rm'] = 'serif'
   mpl.rcParams['figure.dpi'] = 150
   mpl.rcParams['axes.labelsize'] = 'large' # fontsize of the x any y labels
   Pmin = 150 * p.units   ############# upper limit
   n_min = np.argmin(np.abs(p-Pmin))
   Pmin = p[n_min]
   Tmin = tc[n_min]
   TDmin = tdc[n_min]
   Pmed = 500 * p.units   ############# break scale
   n_med = np.argmin(np.abs(p-Pmed))
   Pmed = p[n_med]
   Tmed = tc[n_med]
   TDmed = tdc[n_med]
   Pmax = 1000 * p.units  ############# lower limit

   LG.info('Inside skewt plot')
   #############
   #LG.warning('Interpolating sounding variables')
   Ninterp = 250
   ps = np.linspace(np.max(p),np.min(p), Ninterp)
   ftc = interp1d(p,tc)
   ftdc = interp1d(p,tdc)
   fparcel = interp1d(p,parcel_prof)
   fu = interp1d(p,u)
   fv = interp1d(p,v)
   p = ps
   tc = ftc(ps) * tc.units
   tdc = ftdc(ps) * tdc.units
   parcel_prof = fparcel(ps) * parcel_prof.units
   u = fu(ps) * u.units
   v = fv(ps) * v.units
   n_med = np.argmin(np.abs(p-500*p.units))
   #############
   # Grid plot
   LG.info('creating figure')
   fig = plt.figure(figsize=(11, 13))
   LG.info('created figure')
   LG.info('creating axis')
   gs = gridspec.GridSpec(2, 3, height_ratios=[1,4.2], width_ratios=[6,0.5,1.8])
   fig.subplots_adjust(wspace=0.,hspace=0.)

   # Useful
   bbox_hod_cardinal = dict(ec='none',fc='white', alpha=0.5)
   bbox_barbs = dict(emptybarb=0.075, width=0.1, height=0.2)
   xloc = 0.95
## BOTTOM PLOTS #################################################################
### Main Plot
   # Bottom left plot with the main sounding zoomed in to lower levels.
   LG.info('Creatin SkewT')
   skew_bot = SkewT(fig, rotation=rot, subplot=gs[1,0])
   LG.info('Created SkewT')

   # Plot Data
   ## T and Tdew vs pressure
   skew_bot.plot(p, tc,  'C3')
   skew_bot.plot(p, tdc, 'C0')
   ## Windbarbs
   n = Ninterp//50
   inds, = np.where(p>Pmed)
   inds = inds[::n]
   skew_bot.plot_barbs(pressure=p[inds], u=u[inds], v=v[inds],
                       xloc=xloc, sizes=bbox_barbs)
   ## Parcel profile
   skew_bot.plot(p, parcel_prof, 'k', linewidth=1)
   skew_bot.plot(lcl_p, lcl_t, 'k.')
   skew_bot.plot(cu_base_p, cu_base_t, 'C3o', zorder=100)
   skew_bot.plot(p[0], t0, 'ko', zorder=100)
   skew_bot.plot([p[0].magnitude,lcl_p.magnitude],
                 [tdc[0].magnitude,lcl_t.magnitude],
                 color='C2',ls='--',lw=1, zorder=0)
   ## shade CAPE and CIN
   skew_bot.shade_cape(p, tc, parcel_prof)
   skew_bot.shade_cin(p, tc, parcel_prof, tdc)
   LG.info('plotted CAPE and CIN')
   ## Iso 0
   skew_bot.ax.axvline(0, color='cyan',ls='--',lw=0.65)
   ## Iso t0
   skew_bot.plot([Pmax.magnitude,p[0].magnitude],
                 [t0.magnitude,t0.magnitude],
                 color=(0.5,0.5,0.5),ls='--')
   skew_bot.ax.text(t0,Pmax,f'{t0.magnitude:.1f}°C',va='bottom',ha='left')
   ## Cloud base
   skew_bot.ax.axhline(cu_base_p, color=(0.5,0.5,0.5), ls='--')
   skew_bot.ax.text(cu_base_t,cu_base_p, f'{cu_base_m.magnitude:.0f}m',ha='left')
   LG.info('plotted dew point and sounding')
   ## Dry Adiabats
   LG.info('Plot adiabats, and other grid lines')
   t_dry = units.K * np.arange(243.15, 473.15, 10)
   skew_bot.plot_dry_adiabats(t0=t_dry, linestyles='solid', colors='gray',
                                    linewidth=1)
   ## Moist Adiabats
   t_moist = units.K * np.arange(281.15, 306.15, 4)
   msa = skew_bot.plot_moist_adiabats(t0=t_moist,
                                  linestyles='solid',
                                  colors='lime',
                                  linewidths=.75)
   ## Mixing Ratios
   w = np.array([0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020]).reshape(-1, 1)
   # Vertical extension for the mixing ratio lines
   p_levs = units.hPa * np.linspace(1000, 600, 7)
   skew_bot.plot_mixing_lines(mixing_ratio=w, pressure=p_levs, colors='lime',
                              linestyle='dotted',linewidths=1)

   ## Setup axis
   # Y axis
   skew_bot.ax.set_ylim(Pmax, Pmed)
   # Change pressure labels to height
   yticks, ylabels = [],[]
   for x in reversed(np.arange(0,20000,500)):
      px = m2p(x*units.m)
      if Pmax > px > Pmed:
         yticks.append(px)
         ylabels.append(f'{x:.0f}m\n{px.magnitude:.0f}hPa')
   skew_bot.ax.set_yticks(yticks)
   skew_bot.ax.set_yticklabels(ylabels)
   skew_bot.ax.set_ylabel('Altitude in std atmosphere')
   # X axis
   # Find xlims iteratively since the aspect ratio of the metpy.skewT plots
   # is set to auto. 
   difmin, difmax = 1000,1000
   tmin_old, tmax_old = skew_bot.ax.get_xlim()
   cont = 0
   while cont < 30 and not (difmin < 1e-2 and difmax < 1e-2):
      tmin = np.min([get_bottom_temp(skew_bot.ax,TDmed,Pmed),
                     get_bottom_temp(skew_bot.ax,tdc[0],p[0])])
      tmax = np.max([get_bottom_temp(skew_bot.ax,Tmed,Pmed),
                     get_bottom_temp(skew_bot.ax,tc[0],p[0]),
                     get_bottom_temp(skew_bot.ax,t0,p[0])])
      tmin -= 10   # fine tune
      tmax += 10   #
      difmin = abs(tmin - tmin_old)
      difmax = abs(tmax - tmax_old)
      skew_bot.ax.set_xlim(tmin,tmax)
      tmin_old = tmin
      tmax_old = tmax
      cont += 1
      skew_bot.ax.set_xlabel('Temperature (°C)')

   skew_bot.ax.xaxis.set_major_locator(MultipleLocator(5))
   ## Change the style of the gridlines
   skew_bot.ax.grid(True, which='major', axis='both',
            color='tan', linewidth=1.5, alpha=0.5)

### Clouds
   ax_cloud_bot = plt.subplot(gs[1,1], sharey=skew_bot.ax, zorder=-1)
   ax_cloud_bot.contourf(Xcloud, Ycloud, cloud, cmap='Greys',vmin=0,vmax=1)
   ax_cloud_bot.text(.25,0,'O',ha='center',va='bottom',transform=ax_cloud_bot.transAxes)
   ax_cloud_bot.text(.75,0,'C',ha='center',va='bottom',transform=ax_cloud_bot.transAxes)
   plt.setp(ax_cloud_bot.get_xticklabels(), visible=False)
   plt.setp(ax_cloud_bot.get_yticklabels(), visible=False)
   ax_cloud_bot.set_ylabel('')
   ax_cloud_bot.grid(False)

### Wind Plot
   ax_wind_bot  = plt.subplot(gs[1,2], sharey=skew_bot.ax)
   wspd = np.sqrt(u*u + v*v)
   ax_wind_bot.scatter(wspd, p, c=p, cmap=mcmaps.HEIGHTS, zorder=10)
   ### Background colors ##
   #for i,c in enumerate(mcmaps.WindSpeed.colors):
   #   rect = Rectangle((i*4, 150), 4, 900,  color=c, alpha=0.5,zorder=-1)
   #   ax_wind.add_patch(rect)
   #########################
   # X axis
   ax_wind_bot.set_xlim(0, 56)
   ax_wind_bot.set_xlabel('Wspeed (km/h)')
   ax_wind_bot.xaxis.set_major_locator(MultipleLocator(8))
   ax_wind_bot.xaxis.set_minor_locator(MultipleLocator(4))
   for tick in ax_wind_bot.xaxis.get_major_ticks():
      tick.label.set_fontsize(11)
   plt.setp(ax_wind_bot.get_yticklabels(), visible=False)
   ax_wind_bot.set_ylabel('')
   ax_wind_bot.grid(True, which='minor', axis='x',color=(.8,.8,.8))




## TOP PLOTS ####################################################################
### Sounding upper levels
   # Automatic skew selection to visualize all the data
   isin = False  # Both Td and T are within the drawn limits
   rot = 89
   sign = 1
   delta_rot = 1
   i = 0
   while not isin:
      try: skew_top.ax.remove()
      except UnboundLocalError: pass
      skew_top = SkewT(fig, rotation=rot+sign*i*delta_rot, subplot=gs[0,0])
 
      # Plot Data
      ## T and Tdew vs pressure
      skew_top.plot(p, tc,  'C3')
      skew_top.plot(p, tdc, 'C0')
      ## Parcel profile
      skew_top.plot(p, parcel_prof, 'k', linewidth=1)
      t_top_min, t_top_max = skew_bot.ax.upper_xlim

      ## Setup axis
      skew_top.ax.set_ylim(Pmed, Pmin)
      skew_top.ax.set_xlim(t_top_min, t_top_max)

      T0,T1 = skew_top.ax.upper_xlim
      isin = (T0 <= TDmin.magnitude <= Tmin.magnitude <= T1)
      sign = T0-TDmin.magnitude
      sign /= abs(sign)
      i+= 1
   skew_top.ax.xaxis.set_major_locator(MultipleLocator(5))
   skew_top.ax.set_xlabel('')

   ## Windbarbs
   n = Ninterp//20
   inds, = np.where((p<Pmed) & (p>Pmin))
   inds = inds[::n]
   skew_top.plot_barbs(pressure=p[inds], u=u[inds], v=v[inds],
                       xloc=xloc, sizes=bbox_barbs)
   ## Setup axis
   # Y axis
   # Change pressure labels to height
   yticks_top, ylabels = [],[]
   for x in np.arange(0,20000,1500):
      px = m2p(x*units.m)
      if Pmed > px > Pmin:
         yticks_top.append(px)
         ylabels.append(f'{x:.0f}')
   skew_top.ax.set_yticks(yticks_top)
   skew_top.ax.set_yticklabels(ylabels)
   skew_top.ax.set_ylabel('Altitude (m)')
 
   if len(latlon) > 0:
      skew_top.ax.text(0,1, latlon, va='top', ha='left', color='k',
                     fontsize=12, bbox=dict(boxstyle="round",
                                            ec=None, fc=(1., 1., 1., 0.9)),
                     zorder=100, transform=skew_top.ax.transAxes)
   if len(title) == 0:
      title = f"{(date).strftime('%d/%m/%Y-%H:%M')} (local time)"
   skew_top.ax.set_title(title)

   # Change the style of the gridlines
   skew_top.ax.grid(True, which='major', axis='both', color='tan',
                                         linewidth=1.5, alpha=0.5)
   plt.setp(skew_top.ax.get_xticklabels(), visible=False)

### Clouds
   ax_cloud_top = plt.subplot(gs[0,1], sharey=skew_top.ax, zorder=-1)
   ax_cloud_top.contourf(Xcloud, Ycloud, cloud, cmap='Greys',vmin=0,vmax=1)
   plt.setp(ax_cloud_top.get_xticklabels(), visible=False)
   plt.setp(ax_cloud_top.get_yticklabels(), visible=False)
   ax_cloud_top.set_ylabel('')
   ax_cloud_top.grid(False)

### Wind Plot. Hodograph
   ax_wind_top  = plt.subplot(gs[0,2])
   ax_hod = ax_wind_top
   ax_hod.set_yticklabels([])
   ax_hod.set_xticklabels([])
   L = 80
   bbox = dict(ec='none',fc='white', alpha=0.5)
   ax_hod.text(  0, L-5,'N',   ha='center', va='top',    bbox=bbox_hod_cardinal)
   ax_hod.text(L-5,  0,'E',    ha='right',  va='center', bbox=bbox_hod_cardinal)
   ax_hod.text(-(L-5),0 ,'W',  ha='left',   va='center', bbox=bbox_hod_cardinal)
   ax_hod.text(  0,-(L-5),'S', ha='center', va='center', bbox=bbox_hod_cardinal)
   h = Hodograph(ax_hod) #, component_range=L)
   h.add_grid(increment=20)
   # Plot a line colored by pressure (altitude)
   h.plot_colormapped(-u, -v, p, cmap=mcmaps.HEIGHTS)
   ax_hod.grid(False)
   LG.info('Plotted wind')

## COMMON #######################################################################
   # Ground
   gnd1 = mpcalc.pressure_to_height_std(p[0])
   gnd1 = gnd1.to('m').magnitude
   skew_bot.ax.axhline(p[0],c='k',ls='--')
   ax_cloud_bot.axhline(p[0],c='k',ls='--')
   ax_wind_bot.axhline(p[0],c='k',ls='--')
   trans, _, _ = skew_bot.ax.get_yaxis_text1_transform(0)
   skew_bot.ax.text(0,p[0].magnitude-2,f'GND:{int(gnd1)}m', transform=trans)

## SAVE #########################################################################
   if show: plt.show()
   LG.info('saving')
   fig.savefig(fout, bbox_inches='tight', pad_inches=0.1, dpi=150, quality=90)
   LG.info(f'saved {fout}')
   plt.close('all')

# trans = skew_bot.ax.transScale + skew_bot.ax.transLimits
# print((tdc[n_med].magnitude,Pmed.magnitude), 'in data transforms')
# coors = trans.transform((tdc[n_med].magnitude,Pmed.magnitude))
# print('into',coors, 'in axes')
# print((abs(coors[0]),0), 'in axes transforms')
# tmin = trans.inverted().transform((abs(coors[0]),0))
# print('into',tmin, 'in data')
