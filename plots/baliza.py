#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
HOME = os.getenv('HOME')
STYLE_PATH = os.path.join(here, "styles", "RASP.mplstyle")

import numpy as np
import datetime as dt
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt
plt.style.use(STYLE_PATH)
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from . import utils as ut
# Night
from astral import LocationInfo
from astral.sun import sun

UTCshift = ut.utc_shift()

def rotate_wind(arr):
   return (arr+180) % 360


def night_shade(axs, start, end):
   day = dt.timedelta(days=1)
   # Setup Madrid location
   city = LocationInfo("Madrid", "Spain", "Europe/Madrid", 40.4168, -3.7038)
   # Sunset and sunrise in UTC
   previus_sunset = sun(city.observer, date=start-day)['sunset']
   sunrise = sun(city.observer, date=start)['sunrise']
   sunset = sun(city.observer, date=start)['sunset']
   next_sunrise = sun(city.observer, date=end)['sunrise']
   # Shift to local time
   previus_sunset += UTCshift
   sunrise        += UTCshift
   sunset         += UTCshift
   next_sunrise   += UTCshift

   color = np.array([3, 28, 43,100])/255
   # Suppose you have sunset and sunrise times
   for ax in axs:
      ax.axvspan(previus_sunset, sunrise, color=color)
      ax.axvspan(sunset, next_sunrise, color=color)
   axs[1].text(sunrise,0,'Sunrise')
   axs[1].text(sunset,0,'Sunset')



def compare(obs_df, wrf_df, title='',fout='baliza.png'):
   today = dt.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
   start = today
   end   = start + dt.timedelta(days=1)
   wrf_df = wrf_df[wrf_df.index > today - UTCshift]
   wrf_df = wrf_df.iloc[1:] # skip frist line, pure GFS data not WRF processed
   wrf_df = wrf_df[wrf_df.index < end]
   obs_df = obs_df[obs_df.index > today - UTCshift]


   for df in [wrf_df, obs_df]:
      df['wind_heading_north'] = df['wind_heading'].apply(rotate_wind)
   obs_DF = obs_df.resample("h", label='right', closed='right').mean()

   # Plots
   # Decide fields to plot
   n = 2     # at least always show wind
   include_temp  = obs_df['temperature'].notna().any()
   include_rh    = obs_df['rh'].notna().any()
   include_solar = obs_df['swdown'].notna().any()
   if include_temp:  n += 1
   if include_solar: n += 1
   fig, axes = plt.subplots(n, 1, figsize=(12,4*n), sharex=True,
                                  gridspec_kw={'hspace': 0})
   ax0 = axes[0]
   ax1 = axes[1]

   night_shade(axes, start, end)

   ## Wspd
   # station 5-minute
   x = (obs_df.index + UTCshift).values
   y = obs_df['wind_speed_avg'].values
   ymin = obs_df['wind_speed_min'].values
   ymax = obs_df['wind_speed_max'].values
   ax0.plot(x,y,'C0-.', label='Station full', alpha=.5)
   ax0.fill_between(x,ymin,ymax,color='C0',alpha=.3)
   # station 60-minute
   x = (obs_DF.index + UTCshift).values
   y = obs_DF['wind_speed_avg'].values
   ax0.plot(x,y,'C0-o', label='Station hourly')
   # WRF
   x = (wrf_df.index + UTCshift).values
   y = wrf_df['wind_speed_avg'].values
   ax0.plot(x,y,'C1-o', label='RASP')

   ax0.set_ylabel('Wspeed (km/h)')
   # ax.set_xlabel('Time')
   ax0.set_title(title)
   ax0.set_xticklabels([])


   ## Wdir
   aux = np.concatenate([obs_DF['wind_heading'].values,wrf_df['wind_heading'].values])
   aux = np.max(np.abs(np.diff(aux)))
   if aux > 180:
      center_north = True
      center = 180
   else:
      center_north = False
      center = 0

   if center_north: wdir_col = 'wind_heading_north'
   else: wdir_col = 'wind_heading'
   # station 5-minute
   x = (obs_df.index + UTCshift).values
   y = obs_df[wdir_col].values
   ax1.plot(x,y,'C0-.', label='Station full', alpha=.5)
   # station 60-minute
   x = (obs_DF.index + UTCshift).values
   y = obs_DF[wdir_col].values
   ax1.plot(x,y,'C0-o', label='Station hourly')
   # WRF
   x = (wrf_df.index + UTCshift).values
   y = wrf_df[wdir_col].values
   ax1.plot(x,y,'C1-o', label='RASP')

   # Rotation eye-guides
   ax1.axhline(center,ls='--',color='k')
   ax1.text(end,(center-17)%360,'North',ha='right')

   # Labels
   ax1.set_ylabel('Wdir (°)')
   cardinals_lbl = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
   d_alpha = 360/len(cardinals_lbl)
   cardinals_pos = [(i * d_alpha) % 360 for i in range(len(cardinals_lbl))]
   if center_north: cardinals_pos = [rotate_wind(i) for i in cardinals_pos]
   ax1.set_yticks(cardinals_pos)
   ax1.set_yticklabels(cardinals_lbl)
   ax1.set_ylim(0,360)
   ax1.legend()

   
   msg = dt.datetime.now().strftime('Actualizado: %H:%M %d/%m/%Y')
   props = dict(boxstyle='round', facecolor='white', alpha=0.9)
   ax1.text(1, .998, msg, transform=ax0.transAxes, va='top', ha='right', bbox=props)
   
   n_ax = 2  # after plotting wind, the next available axis is axes[n_ax]
   if include_temp:
      # Temperature
      ax2 = axes[n_ax]
      n_ax += 1  # increase counting for next plot
      ax2_twin = ax2.twinx()
      x = (obs_df.index + UTCshift).values
      y = obs_df['temperature'].values
      ax2.plot(x,y, 'C0-.', alpha=.5)
      x = (obs_DF.index + UTCshift).values
      y = obs_DF['temperature'].values
      ax2.plot(x,y, 'C0-o')
      x = (wrf_df.index + UTCshift).values
      y = wrf_df['temperature'].values
      ax2.plot(x,y, 'C1-o')

      if include_rh:
         # Relative humidity
         x = (obs_df.index + UTCshift).values
         y = obs_df['temperature'].values
         ax2_twin.plot(x,y, 'C2-.', alpha=.5)
         x = (obs_DF.index + UTCshift).values
         y = obs_DF['temperature'].values
         ax2_twin.plot(x,y, 'C2-o', label='Station Rh (%)')
         x = (wrf_df.index + UTCshift).values
         y = wrf_df['temperature'].values
         ax2_twin.plot(x,y, 'C3-o', label='RASP Rh (%)')
         # Settings
         ax2.set_ylabel('Temperature (°C)')
         # ax2_twin.set_ylabel('Relative Humidity (%)')
         ax2_twin.legend(loc=2)
         ax2_twin.set_ylim(0,100)
   if include_solar:
      ax3 = axes[n_ax]
      n_ax += 1  # increase counting for next plot
      x = (obs_df.index + UTCshift).values
      y = obs_df['swdown'].values
      ax3.plot(x,y, 'C0-.', alpha=.5)
      x = (obs_DF.index + UTCshift).values
      y = obs_DF['swdown'].values
      ax3.plot(x,y, 'C0-o')
      x = (wrf_df.index + UTCshift).values
      y = wrf_df['swdown'].values
      ax3.plot(x,y, 'C1-o')
      # Settings
      ax3.set_ylabel('Solar ($W/m^2$)')

   # Grid and ticks
   for ax in axes:
      ax.grid()
      ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
      ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
      # Grid lines ONLY for X-axis minor ticks
      ax.xaxis.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
      ax.set_xlim(start,end)
   axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
   for label in ax1.get_xticklabels(which='major'):
      label.set(rotation=20, horizontalalignment='right')

   fig.tight_layout()
   fig.savefig(fout)
 
