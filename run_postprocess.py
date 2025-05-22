#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Sys modules
from pathlib import Path
import os, sys, argparse
here = os.path.dirname(os.path.realpath(__file__))
HOME = os.getenv('HOME')

# Loggings
import logging
import log_help
LG = logging.getLogger("main")
LGp = logging.getLogger("perform")

# RASP modules
import pandas as pd
import utils as ut
from calc_data import CalcData
# stations
from stations.extract_wrf import save_prediction
# meteograms
from meteogram_writer import make_meteogram_timestep, append_to_meteogram
# web, sounding & meteogram
from plots.web import generate_background, generate_scalars, generate_vectors
from plots.sounding import skew_t_plot
from plots.meteogram import plot_meteogram


def existing_file(path):
   """Helper function for argparse to check if input file exists"""
   if not os.path.isfile(path):
      raise argparse.ArgumentTypeError(f"File not found: {path}")
   return path

def parse_args():
   """
   Define input options
   - filepath: wrfout file to process
   - config: path to config.ini file
   """
   parser = argparse.ArgumentParser(
       description="Post-process a WRF output file and generate plots." )
   parser.add_argument("filepath",  type=existing_file,
          help="Path to the WRF NetCDF file (wrfout_<domain>_<date>)" )
   parser.add_argument("--config", default=f"{here}/config.ini",
                       help=f"Path to config.ini (default: {here}/config.ini)")
   return parser.parse_args()

@log_help.timer(LG,LGp)
def process_file(fname, configfile, LG):
   """
   Run post-processing for wrfout file
   """
   LG.info(f"Processing file: {fname}")
   paths = ut.load_config_or_die(configfile)
   output_folder  = paths['wrfout_folder']
   plots_folder   = paths['plots_folder']
   data_folder    = paths['data_folder']
   configs_folder = paths['configs_folder']
   plots_ini      = paths['plots_ini']
   zooms_ini      = paths['zooms_ini']

   A = CalcData(fname, OUT_folder=plots_folder, DATA_folder=data_folder)
   domain = A.meta['domain']
   zooms = ut.load_zooms(zooms_ini, domain=domain)

   # Read station metadata file
   stations_csv = Path(configs_folder) / f"stations_{domain}.csv"
   if not stations_csv.exists():
      LG.warning(f"Station list not found: {stations_csv}")
   else:
      LG.info(f"Reading station list from: {stations_csv}")
      df = pd.read_csv(stations_csv)
      predictions_folder = A.paths["data_stations"] / "predictions"
      ut.check_directory(predictions_folder)  # create if missing
      for i, row in df.iterrows():
         # try:
            lat, lon = float(row["lat"]), float(row["lon"])
            station_id = str(row["name"]).strip()
            LG.info(f"Saving prediction for station '{station_id}'")
            save_prediction(A, station_id, lat, lon, predictions_folder)
         # except Exception as e:
         #    LG.exception(f"Failed to process station '{row}': {e}")

   # Background (it will skip if the files already exist)
   LG.info("Generating background maps...")
   generate_background(A.paths['plots_common'], A.geometry,
                       csv_dir=configs_folder, zooms=zooms)

   # Scalars: T2, RH, Wstar, CAPE, etc
   LG.info("Plotting scalar fields...")
   generate_scalars(A, config_path=plots_ini, zooms=zooms)

   # Vectors: winds, streamlines, etc
   LG.info("Plotting vector fields...")
   generate_vectors(A, config_path=plots_ini, zooms=zooms)

   # Plot soundings and meteograms
   LG.info("Plotting soundings and meteograms")
   soundings_csv = Path(configs_folder) / f"soundings_{domain}.csv"
   if not soundings_csv.exists():
      LG.warning(f"Soundings file missing: {soundings_csv}")
   df = pd.read_csv(f"{configs_folder}/soundings_{domain}.csv", 
                    names=['lat', 'lon', 'code', 'place'])
   for _, row in df.iterrows():
      lat,lon, code,place = row['lat'],row['lon'], row['code'],row['place']

      # Sounding
      fout = A.paths["plots_daily"] / f"{A.tail_h}_sounding_{code}.webp"
      skew_t_plot(A, lat, lon, name=place, fout=fout)

      # Meteogram
      day_nc = A.paths["data_meteograms"] / f"meteogram_{code}.nc"
      ds = make_meteogram_timestep(A, lat, lon)
      ds_full = append_to_meteogram(ds, day_nc)
      if len(ds_full["time"]) >= 2:
         fout = A.paths["plots_daily"] / f"meteogram_{code}.webp"
         plot_meteogram(day_nc, name=place, fout=fout)
      else:
         LG.debug(f"Skipping meteogram plot for {code} (only one time point)")

   LG.info(f"Finished processing {fname}")

def main():
   args = parse_args()
   fname = args.filepath
   fname = Path(fname)
   config_file = args.config

   # Get common variables for setting up LOG
   is_cron = bool(os.getenv('RUN_BY_CRON'))
   domain = ut.get_domain(fname)
   date = ut.file2date(fname)

   # Prepare standard GFSbatch path
   batch_path = fname.parent / "batch.txt"
   batch = ut.get_GFSbatch(batch_path)
   script_path = os.path.realpath(__file__)

   LG, LGp = log_help.batch_logger(script_path, domain, batch,
                                   is_cron, log_dir='logs')

   LG.info("=================================================")
   LG.info("=                New run started                =")
   LG.info("=================================================")
   LG.info(f"Cron: {is_cron}")
   try:
      process_file(fname, config_file, LG)
   except Exception as e:
      LG.exception(f"Failed to process file {fname}: {e}")
      sys.exit(1)

if __name__ == "__main__":
   main()
