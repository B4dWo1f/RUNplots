#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import log_help
import logging
LG = logging.getLogger(f'main.{__name__}')
LGp = logging.getLogger(f'perform.{__name__}')

import requests
import datetime as dt
from bs4 import BeautifulSoup
import pandas as pd
import re
import stations.utils as ut

UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))

def download_data(url, Ndays=3):
   # Define the URL
   hours = 24  # or 1, 0.01, etc.
   url = f'https://freeflightwx.com/penanegra/table.php?h={hours}'

   # Request the page
   response = requests.get(url)
   response.raise_for_status()  # raise error if request faile

   # Parse HTML with BeautifulSoup
   soup = BeautifulSoup(response.text, 'html.parser')
   table = soup.find('table', class_='db-table')
   rows = table.find_all('tr')

   # Collect data
   data = []

   now = dt.datetime.now().replace(microsecond=0, second=0)
   current_date = now.date()
   previous_time = None
   # midnight_crossed = False
   for row in rows[1:]:  # Skip header
      cols = [td.get_text(strip=True) for td in row.find_all('td')]
      if len(cols) < 14: continue  # skip malformed rows

      # Parse time string
      time_str = cols[1]
      time_obj = dt.datetime.strptime(time_str, "%H:%M:%S").time()

      # Detect date change (crossed midnight)
      if previous_time and time_obj > previous_time:
         # midnight_crossed = True
         current_date -= dt.timedelta(days=1)
      previous_time = time_obj
      full_datetime = dt.datetime.combine(current_date, time_obj)
      full_datetime = full_datetime - UTCshift
      # LG.critical(f'Original time: {full_datetime}')
      # LG.critical(f'UTC time?: {full_datetime - UTCshift}')

      # Extract heading using regex
      heading_match = re.search(r'\((\d+)\)', cols[5])
      heading_deg = int(heading_match.group(1)) if heading_match else None

      entry = {'time': full_datetime, #cols[1],
               'wind_speed_avg': float(cols[2]) * 1.60934,
               'wind_speed_max': float(cols[3]) * 1.60934,
               'wind_speed_min': float(cols[4]) * 1.60934,
               'wind_heading': heading_deg,
               'pressure': float(cols[7]) / 100.0,  # QNH in hPa
               'temperature': float(cols[8]),
               'rh': int(cols[9]),
               'dew': float(cols[10])
               }
      data.append(entry)

   # Convert to DataFrame
   df = pd.DataFrame(data)
   df = ut.reconcile_station_dataframe(df)
   return df
