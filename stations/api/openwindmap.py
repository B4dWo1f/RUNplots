#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import json
import pandas as pd
import datetime as dt
from stations.schema import STATION_CSV_COLUMNS
import stations.utils as ut

UTCshift = dt.datetime.now() - dt.datetime.utcnow()
UTCshift = dt.timedelta(hours = round(UTCshift.total_seconds()/3600))

def download_data(url, Ndays=3):
   station_id = url.split('/')[-1].split('-')[-1]
   # LG.info(f"Updating station {station_id} into file {fname}")
   today = dt.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
   today = pd.Timestamp( today - UTCshift )

      # Format the start date for API call
   start = dt.datetime.now() - dt.timedelta(days=Ndays)
   start = start.strftime('%Y-%m-%d')
   stop = 'now'
   fmt = 'json'

   # Download the data via curl
   url_base = f"http://api.pioupiou.fr/v1/archive/{station_id}"
   url = f"{url_base}?start={start}&stop={stop}&format={fmt}"
   com = f"curl -s '{url}'"
   # LG.debug(f'CURL command: {com}')
   data = os.popen(com).read().strip()

   try:
      data = json.loads(data)
      LG.debug("JSON data successfully parsed.")
   except json.JSONDecodeError:
      LG.error("Failed to parse JSON data from API.")
      return

   # Convert json data to csv
   columns = data.get('legend', [])
   rows = data.get('data', [])

   if not rows:
      LG.warning("No data returned from API.")
      return

   # Create a DataFrame and parse the timestamp
   df = pd.DataFrame(rows, columns=columns)
   fmt_date = '%Y-%m-%dT%H:%M:%S.%fZ'
   df['time'] = pd.to_datetime(df['time'], format=fmt_date)
   # df.set_index('time', inplace=True)
   # df = df[df.index > today]  #.to_csv(fname, float_format='%.3f')
   df = df[df['time'] > today]

   df = ut.reconcile_station_dataframe(df)
   return df

