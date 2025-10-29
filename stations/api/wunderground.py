#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import logging
import datetime as dt
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from stations.schema import STATION_CSV_COLUMNS
# from .. import utils  as ut
import stations.utils as sut
import utils as ut   # root level

LG = logging.getLogger(f"main.{__name__}")

UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))

DIR_TO_DEGREES = {
    'North': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5, 'East': 90.0,
    'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5, 'South': 180.0, 'SSW': 202.5,
    'SW': 225.0, 'WSW': 247.5, 'West': 270.0, 'WNW': 292.5, 'NW': 315.0,
    'NNW': 337.5, '': None
}


def feet2m(l):
   return l*0.3048
def farenheit2celsius(t):
   return (t-32)*5/9
def mph2kmh(v):
   return v*1.60934
def in2hpa(p):
   return p*33.863889532610884



def parse_location_info(soup):
   heading = soup.find('div', {'class': 'heading'}).text.replace('info', '')
   name, station_id = [x.strip() for x in heading.split('-')]

   info = soup.find('div', {'class': 'sub-heading'}).text.split()
   _, elev_ft, _, lat, lat_dir, lon, lon_dir = info
   lat = float(lat) * (1 if lat_dir.startswith('N') else -1)
   lon = float(lon) * (1 if lon_dir.startswith('E') else -1)
   elev = feet2m(float(elev_ft))

   return name, station_id, lat, lon, elev

def parse_weather_table(table, fill_min=True): #, date):
   today = dt.datetime.now().date()
   data = []
   rows = table.find_all('tr')[2:]  # Skip headers
   for row in rows:
      data_row = [col.text for col in row.find_all('td')]
      if len(data_row) < 12: continue
      try:
         time_str,temp_f,dew_f,rh,winddir,wspd,gust,press_in,\
                                                 _,_,_,solar = data_row
         time = dt.datetime.strptime(time_str, '%I:%M %p').time()
         timestamp = dt.datetime.combine(today, time)
         timestamp -= UTCshift   # make it UTC time
         #XXX fill in wind_speed_min
         wspd = mph2kmh(float(wspd.split()[0]))
         gust = mph2kmh(float(gust.split()[0]))
         if fill_min: wmin = wspd - np.abs((gust-wspd))
         else: wmin = np.nan
         solar = float(solar.split()[0])
         data.append({
             'time': timestamp,
             'temperature': farenheit2celsius(float(temp_f.split()[0])),
             'rh': float(rh.split()[0]),
             'wind_heading': DIR_TO_DEGREES.get(winddir, np.nan),
             'wind_speed_avg': wspd,
             'wind_speed_min': wmin,
             'wind_speed_max': gust,
             'pressure': in2hpa(float(press_in.split()[0])),
             'swdown': solar,
         })
      except Exception as e:
          LG.warning(f"Row parsing failed: {e}")
          continue
   return pd.DataFrame(data)

def download_data(url_base):
   """Downloads and parses data from a Wunderground station HTML page."""
   date = dt.datetime.now() #- dt.timedelta(days=1)
   date_str = date.strftime('%Y-%m-%d')
   url = f"{url_base}/{date_str}/{date_str}/daily"
   LG.info(f'wunderground: {url}')
   html = sut.make_request(url, "table.history-table.desktop-table")
   soup = BeautifulSoup(html, 'html.parser')

   name, station_id, lat, lon, _ = parse_location_info(soup)
   table = soup.find('table', {'class': 'history-table desktop-table'})
   if table is None:
       raise ValueError("Failed to find weather data table in the HTML")

   # today = dt.datetime.utcnow().date()
   df = parse_weather_table(table) #, today)
   if df.empty:
       raise ValueError("Parsed weather table is empty")

   # Add metadata
   df["station_id"] = station_id
   df["lat"] = lat
   df["lon"] = lon
   df = df[STATION_CSV_COLUMNS]
   df.set_index("time", inplace=True)
   return df
