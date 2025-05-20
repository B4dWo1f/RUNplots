#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from pathlib import Path
here = Path(__file__).resolve().parent
import logging
import sys

import stations.utils as sut
from stations.utils import save_station_csv, validate_station_df
from stations.schema import STATION_CSV_COLUMNS
from stations.api import openwindmap, wunderground
import utils as ut
from plots.baliza import compare

# Setup logger
LG = logging.getLogger("download_api")
logging.basicConfig(
   level=logging.INFO,
   format='[%(asctime)s] %(levelname)s - %(message)s',
   datefmt='%Y/%m/%d %H:%M:%S'
)


# Select the correct API backend based on URL
def choose_backend(url):
   """
   Returns the appropriate API download function based on station URL.

   Args:
       url (str): The URL associated with the station.
   Returns:
       Callable: Function to download station data.

   Raises:
       ValueError: If the URL's provider is not supported.
   """
   if "openwindmap.org" in url: return openwindmap.download_data
   elif "wunderground.com" in url: return wunderground.download_data
   # Add more providers here as needed
   else: raise ValueError(f"Unsupported API for: {url}")


def download():
   """
   Download station observation data for today and store it in the
   appropriate observations folder.
   """
   configfile = here/'config.ini'  #XXX dishonor in your cow!
   paths = ut.load_config_or_die(configfile)

   for domain in ['d01','d02']:
      STATIONS_CSV = paths['configs_folder'] / f"stations_{domain}.csv"
      OUT_DIR = paths['data_folder'] / "stations/observations"
      ut.check_directory(OUT_DIR)

      if not STATIONS_CSV.exists():
         LG.critical(f"Missing config file: {CONFIG_CSV}")
         sys.exit(1)

      df = pd.read_csv(STATIONS_CSV)

      for _, row in df.iterrows():
         name = str(row["name"]).strip()
         url  = str(row["url"]).strip()

         try:
            backend = choose_backend(url)
            data_df = backend(url)
            save_station_csv(data_df, OUT_DIR / f"{name}.csv")
            LG.info(f"Saved {len(data_df)} obs rows for {name}")
         except Exception as e:
            LG.error(f"Failed to fetch or save station {name}: {e}")

def compare_prediction_observation_dirs(pred_dir, obs_dir):
   """
   Compares files in predictions and observations folders and logs
   discrepancies

   Args:
       pred_dir (str or Path): Path to predictions folder.
       obs_dir (str or Path): Path to observations folder.

   Returns:
       Set[str]: List of common CSV files.
   """
   pred_dir = Path(pred_dir).expanduser().resolve()
   obs_dir  = Path(obs_dir).expanduser().resolve()

   pred_files = {f.name for f in pred_dir.glob("*.csv")}
   obs_files  = {f.name for f in obs_dir.glob("*.csv")}

   only_in_pred = pred_files - obs_files
   only_in_obs  = obs_files - pred_files
   common_files = pred_files & obs_files

   for f in sorted(only_in_pred):
      LG.warning(f"File present in predictions only: {f}")
   for f in sorted(only_in_obs):
      LG.warning(f"File present in observations only: {f}")

   if not only_in_pred and not only_in_obs:
      LG.info("Prediction and observation folders are synchronized.")
   return common_files

def plot():
   """
   Generate comparison plots for all stations that exist in both folders.
   """
   configfile = 'config.ini'
   paths = ut.load_config_or_die(configfile)
   PLOTS = paths['plots_stations']
   PREDICTIONS = paths['data_folder'] / "stations/predictions"
   OBSERVATIONS = paths['data_folder'] / "stations/observations"
   common = compare_prediction_observation_dirs(PREDICTIONS, OBSERVATIONS)
   for fname in sorted(common):
      pred = PREDICTIONS  / fname
      try:
         obs  = OBSERVATIONS / fname
         fout = PLOTS / fname.replace('csv','webp')
         df_pred = sut.read_station_csv(pred)
         df_obs  = sut.read_station_csv(obs)
         title = f"Baliza {fout.stem.capitalize()}"
         compare(df_obs, df_pred, title=title, fout=fout)
      except Exception as e:
         LG.error(f"Failed to plot {fname}: {e}")


def main():
   download()
   plot()

if __name__ == "__main__":
   main()
