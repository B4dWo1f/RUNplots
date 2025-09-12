#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import log_help
import logging
LG = logging.getLogger(f'main.{__name__}')
LGp = logging.getLogger(f'perform.{__name__}')

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stations.schema import STATION_CSV_COLUMNS
from playwright.sync_api import sync_playwright


def make_request(url: str, element: str = None) -> str:
   t_out = 15000
   with sync_playwright() as p:
      browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
      context = browser.new_context()
      try:
         page = browser.new_page()
         page.goto(url, timeout=t_out)  # 15 seconds
         # Wait for a specific element if provided
         if element:
            page.wait_for_selector(element, timeout=t_out)
         return page.content()
      finally:
         context.close()
         browser.close()


def validate_station_df(df):
   """
   Check that the provided DataFrame contains all the requested columns
   Added an exception in case 'time' was passed as index instead of column
   """
   if df.index.name == 'time':
      req_columns = [c for c in STATION_CSV_COLUMNS if c !='time']
      LG.warning('time column was passed as index')
   else: req_columns = STATION_CSV_COLUMNS
   missing = set(req_columns) - set(df.columns)
   if missing:
      raise ValueError(f"Missing columns in station CSV: {missing}")
   else: return True


def save_station_csv(df, csv_path):
   """
   Save a cleaned station DataFrame to CSV. Overwrites existing file.

   Args:
       df (pd.DataFrame): DataFrame to save
       csv_path (Path or str): Output path
   """
   try:
      if validate_station_df(df):
         df.sort_values("time", inplace=True)
         df.round(5).to_csv(csv_path)
         LG.info(f"Saved station data to {csv_path}")
      else: LG.critical('DataFrame not valid')
   except Exception as e:
      LG.critical(f"Failed to save {csv_path}: {e}")
      raise


def read_station_csv(csv_path):
   """
   Load a station CSV file, validate its structure and types.

   Args:
       csv_path (str or Path): Path to CSV file.

   Returns:
       pd.DataFrame: Cleaned and validated DataFrame.

   Raises:
       FileNotFoundError: If file doesn't exist.
       ValueError: If required columns are missing or malformed.
   """
   csv_path = Path(csv_path)
   if not csv_path.is_file():
      raise FileNotFoundError(f"Station CSV not found: {csv_path}")

   try: 
      df = pd.read_csv(csv_path, parse_dates=["time"])
      df.set_index('time', inplace=True)
   except Exception as e:
      raise ValueError(f"Failed to read or parse CSV: {e}")

   # Ensure correct dtypes (date already parsed)
   for col in ["wind_speed_min", "wind_speed_avg", "wind_speed_max",
               "wind_heading", "temperature", "rh", "pressure"]:
      if col in df.columns:
         df[col] = pd.to_numeric(df[col], errors='coerce')

   # # Optional: drop rows with invalid dates or NaNs in critical fields
   # df.dropna(subset=["time", "lat", "lon"], inplace=True)
   return df #[STATION_CSV_COLUMNS]  # Enforce column order


def reconcile_station_dataframe(df):
   """
   Ensure the dataframe matches the REQUIRED_COLUMNS format.
   Fills missing columns with NaN and reorders appropriately.
   """
   df = df.copy()

   # Use numpy.nan for missing values
   df = df.mask(df.isnull(), np.nan)

   # Drop columns not in schema
   df = df[[col for col in df.columns if col in STATION_CSV_COLUMNS]]

   # Add missing columns as NaN
   for col in STATION_CSV_COLUMNS:
      if col not in df.columns:
         df[col] = np.nan

   # Reorder
   df = df[STATION_CSV_COLUMNS]
   df.set_index('time', inplace=True)
   if not validate_station_df(df):
      LG.critical('Invalid df formed')
      sys.exit(1)
   return df

