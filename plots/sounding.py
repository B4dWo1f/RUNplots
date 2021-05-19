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

@log_help.timer(LG)
def skewt_plot(p,tc,tdc,t0,date,u=None,v=None,fout='sounding.png',latlon='',title='',show=False):
   """
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
   Pmin = 150    # XXX upper limit
   Pmax = 1000   # XXX lower limit
   LG.debug('Checking units')
   if p.attrs['units'] != 'hPa':
      LG.critical('P wrong units')
      exit()
   if tc.attrs['units'] != 'degC':
      LG.critical('Tc wrong units')
      exit()
   if tdc.attrs['units'] != 'degC':
      LG.critical('Tdc wrong units')
      exit()
   if t0.attrs['units'] != 'degC':
      LG.critical('T0 wrong units')
      exit()
   if type(u) != type(None) and type(v) != type(None):
      if u.attrs['units'] != 'm s-1':
         LG.critical('Wind wrong units')
         exit()
   LG.info('Inside skewt plot')
   p = p.values
   tc = tc.values
   tdc = tdc.values
   t0 = t0.mean().values
   u = u.values * 3.6  # km/h
   v = v.values * 3.6  # km/h
   ############
   LG.warning('Interpolating sounding variables')
   Ninterp = 250
   ps = np.linspace(np.max(p),np.min(p), Ninterp)
   ftc = interp1d(p,tc)
   ftdc = interp1d(p,tdc)
   fu = interp1d(p,u)
   fv = interp1d(p,v)
   p = ps
   tc = ftc(ps)
   tdc = ftdc(ps)
   u = fu(ps)
   v = fv(ps)
   ############
   # Grid plot
   LG.info('creating figure')
   fig = plt.figure(figsize=(11, 12))
   LG.info('created figure')
   LG.info('creating axis')
   gs = gridspec.GridSpec(1, 3, width_ratios=[6,0.4,1.8])
   fig.subplots_adjust(wspace=0.,hspace=0.)
   # ax1 = plt.subplot(gs[1:-1,0])
   # Adding the "rotation" kwarg will over-ride the default MetPy rotation of
   # 30 degrees for the 45 degree default found in NCL Skew-T plots
   LG.info('created axis')
   LG.info('Creatin SkewT')
   skew = SkewT(fig, rotation=45, subplot=gs[0,0])
   ax = skew.ax
   LG.info('Created SkewT')

   if len(latlon) > 0:
       ax.text(0,1, latlon, va='top', ha='left', color='k',
                     fontsize=12, bbox=dict(boxstyle="round",
                                            ec=None, fc=(1., 1., 1., 0.9)),
                     zorder=100, transform=ax.transAxes)
   # Plot the data, T and Tdew vs pressure
   skew.plot(p, tc,  'C3')
   skew.plot(p, tdc, 'C0')
   LG.info('plotted dew point and sounding')

   # LCL
   lcl_p, lcl_t = mpcalc.lcl(p[0]*units.hPa,
                                              t0*units.degC,
                                              tdc[0]*units.degC)
   skew.plot(lcl_p, lcl_t, 'k.')
   lcl_p = lcl_p.magnitude
   lcl_t = lcl_t.magnitude

   # Calculate the parcel profile  #XXX units workaround
   parcel_prof = mpcalc.parcel_profile(p* units.hPa,
                                       t0 * units.degC,
                                       tdc[0]* units.degC).to('degC')
   # Plot the parcel profile as a black line
   skew.plot(p, parcel_prof, 'k', linewidth=1)
   LG.info('plotted parcel profile')

   # Plot cloud base
   # p_base, t_base = find_cross(parcel_prof.magnitude, tc, p, tc, interp=True)
   # p_base = np.max([lcl_pressure.magnitude, p_base])
   # t_base = np.max([lcl_temperature.magnitude, t_base])
   p_base, t_base = wrf_calcs.post_process.get_cloud_base(parcel_prof, p, tc, lcl_p, lcl_t)
   parcel_cross = p_base, t_base
   m_base = mpcalc.pressure_to_height_std(np.array(p_base)*units.hPa)
   m_base = m_base.to('m').magnitude
   skew.plot(p_base, t_base, 'C3o', zorder=100)
   skew.ax.text(t_base, p_base, f'{m_base:.0f}m',ha='left')

   # shade CAPE and CIN
   skew.shade_cape(p* units.hPa,
                   tc * units.degC, parcel_prof)
   skew.shade_cin(p * units.hPa,
                  tc * units.degC,
                  parcel_prof,
                  tdc * units.degC)
   LG.info('plotted CAPE and CIN')

   ## Clouds ############
   ps = np.linspace(np.max(p),np.min(p),500)
   tcs = interp1d(p,tc)(ps)
   tds = interp1d(p,tdc)(ps)
   x0 = 0.3
   t = 0.2
   overcast = wrf_calcs.util.fermi(tcs-tds, x0=x0,t=t)
   overcast = overcast/wrf_calcs.util.fermi(0, x0=x0,t=t)
   cumulus = np.where((p_base>ps) & (ps>parcel_cross[0]),1,0)
   cloud = np.vstack((overcast,cumulus)).transpose()
   ax_cloud = plt.subplot(gs[0,1],sharey=ax, zorder=-1)
   ax_cloud.imshow(cloud,origin='lower',extent=[0,2,p[0],p[-1]],aspect='auto',cmap='Greys',vmin=0,vmax=1)
   ax_cloud.text(0,0,'O',transform=ax_cloud.transAxes)
   ax_cloud.text(0.5,0,'C',transform=ax_cloud.transAxes)
   ax_cloud.set_xticks([])
   plt.setp(ax_cloud.get_yticklabels(), visible=False)
   #####################

   if type(u) != type(None) and type(v) != type(None):
      LG.info('Plotting wind')
      ax_wind = plt.subplot(gs[0,2], sharey=ax, zorder=-1)
      ax_wind.yaxis.tick_right()
      # ax_wind.xaxis.tick_top()
      wspd = np.sqrt(u*u + v*v)
      ax_wind.scatter(wspd, p, c=p, cmap=mcmaps.HEIGHTS, zorder=10)
      gnd = mpcalc.pressure_to_height_std(np.array(p[0])*units.hPa)
      gnd = gnd.to('m')
      ### Background colors ##
      #for i,c in enumerate(mcmaps.WindSpeed.colors):
      #   rect = Rectangle((i*4, 150), 4, 900,  color=c, alpha=0.5,zorder=-1)
      #   ax_wind.add_patch(rect)
      #########################
      # X axis
      ax_wind.set_xlim(0,56)
      ax_wind.set_xlabel('Wspeed (km/h)')
      ax_wind.set_xticks([0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56])
      ax_wind.set_xticklabels(['0','','8','','16','','24','32','40','48','56'], fontsize=11, va='center')
      # Y axis
      plt.setp(ax_wind.get_yticklabels(), visible=False)
      # ax_wind.grid()
      def p2h(x):
         """
         x in hPa
         """
         y = mpcalc.pressure_to_height_std(np.array(x)*units.hPa)
         # y = y.metpy.convert_units('m')
         y = y.to('m')
         return y.magnitude
      def h2p(x):
         """
         x in m
         """
         # x = x.values
         y = mpcalc.height_to_pressure_std(np.array(x)*units.m)
         # y = y.metpy.convert_units('hPa')
         y = y.to('hPa')
         return y.magnitude

      # Duplicate axis in meters
      ax_wind_m = ax_wind.secondary_yaxis(1.02,functions=(p2h,h2p))
      ax_wind_m.set_ylabel('height (m)')
      # XXX Not working
      ax_wind_m.yaxis.set_major_formatter(ScalarFormatter())
      ax_wind_m.yaxis.set_minor_formatter(ScalarFormatter())
      #################
      ax_wind_m.set_color('red')
      ax_wind_m.tick_params(colors='red',size=7, width=1, which='both')  # 'both' refers to minor and major axes
      # Hodograph
      ax_hod = inset_axes(ax_wind, '110%', '30%', loc=1)
      ax_hod.set_yticklabels([])
      ax_hod.set_xticklabels([])
      L = 80
      ax_hod.text(  0, L-5,'N', horizontalalignment='center',
                               verticalalignment='center')
      ax_hod.text(L-5,  0,'E', horizontalalignment='center',
                               verticalalignment='center')
      ax_hod.text(-(L-5),0 ,'W', horizontalalignment='center',
                               verticalalignment='center')
      ax_hod.text(  0,-(L-5),'S', horizontalalignment='center',
                               verticalalignment='center')
      h = Hodograph(ax_hod, component_range=L)
      h.add_grid(increment=20)
      # Plot a line colored by pressure (altitude)
      h.plot_colormapped(-u, -v, p, cmap=mcmaps.HEIGHTS)
      LG.info('Plotted wind')

      ## Plot only every n windbarb
      n = Ninterp//30
      inds, = np.where(p>Pmin)
      # break_p = 25
      # inds_low = inds[:break_p]
      # inds_high = inds[break_p:]
      # inds = np.append(inds[::n], inds_high)
      inds = inds[::n]
      skew.plot_barbs(pressure=p[inds], # * units.hPa,
                      u=u[inds],
                      v=v[inds],
                      xloc=0.985, # fill_empty=True,
                      sizes=dict(emptybarb=0.075, width=0.1, height=0.2))

   # Add relevant special lines
   ## cumulus base
   skew.ax.axhline(p_base, color=(0.5,0.5,0.5), ls='--')
   ax_wind.axhline(p_base,c=(0.5,0.5,0.5),ls='--')
   ax_cloud.axhline(p_base,c=(0.5,0.5,0.5),ls='--')
   ## cumulus top
   skew.ax.axhline(parcel_cross[0], color=(.75,.75,.75), ls='--')
   ax_wind.axhline(parcel_cross[0], color=(.75,.75,.75), ls='--')
   ax_cloud.axhline(parcel_cross[0], color=(.75,.75,.75), ls='--')

   # Ground
   skew.ax.axhline(p[0],c='k',ls='--')
   ax_wind.axhline(p[0],c='k',ls='--')
   ax_cloud.axhline(p[0],c='k',ls='--')
   ax_wind.text(56,p[0],f'{int(gnd.magnitude)}m',horizontalalignment='right')
   # Choose starting temperatures in Kelvin for the dry adiabats
   LG.info('Plot adiabats, and other grid lines')
   skew.ax.text(t0,Pmax,f'{t0:.1f}°C',va='bottom',ha='left')
   skew.ax.axvline(t0, color=(0.5,0.5,0.5),ls='--')
   t0 = units.K * np.arange(243.15, 473.15, 10)
   skew.plot_dry_adiabats(t0=t0, linestyles='solid', colors='gray', linewidth=1)

   # Choose temperatures for moist adiabats
   t0 = units.K * np.arange(281.15, 306.15, 4)
   msa = skew.plot_moist_adiabats(t0=t0,
                                  linestyles='solid',
                                  colors='lime',
                                  linewidths=.75)

   # Choose mixing ratios
   w = np.array([0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020]).reshape(-1, 1)

   # Choose the range of pressures that the mixing ratio lines are drawn over
   p_levs = units.hPa * np.linspace(1000, 400, 7)
   skew.plot_mixing_lines(mixing_ratio=w, pressure=p_levs,
                          linestyle='dotted',linewidths=1, colors='lime')

   LG.info('Plotted adiabats, and other grid lines')

   skew.ax.set_ylim(Pmax, Pmin)
   skew.ax.set_xlim(-20, 43)

   # Change the style of the gridlines
   ax.grid(True, which='major', axis='both',
            color='tan', linewidth=1.5, alpha=0.5)

   ax.set_xlabel("Temperature (C)")
   ax.set_ylabel("P (hPa)")
   if len(title) == 0:
      title = f"{(date).strftime('%d/%m/%Y-%H:%M')} (local time)"
   ax.set_title(title, fontsize=20)
   if show: plt.show()
   LG.info('saving')
   fig.savefig(fout, bbox_inches='tight', pad_inches=0.1, dpi=150, quality=90)
   LG.info('saved')
   plt.close('all')
