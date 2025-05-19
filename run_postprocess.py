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
from meteogram_writer import make_meteogram_timestep, append_to_meteogram
from plots.sounding import skew_t_plot
from plots.meteogram import plot_meteogram
from plots.web import generate_background, generate_scalars, generate_vectors


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
   soundings_csv = f"{configs_folder}/soundings_{domain}.csv"
   if not os.path.isfile(soundings_csv):
      LG.warning(f"Soundings file missing: {soundings_csv}")
   df = pd.read_csv(f"{configs_folder}/soundings_{domain}.csv", 
                    names=['lat', 'lon', 'code', 'place'])
   for _, row in df.iterrows():
      lat,lon, code,place = row['lat'],row['lon'], row['code'],row['place']

      # Sounding
      fout = A.paths["plots_daily"] / f"{A.tail_h}_sounding_{code}.png"
      skew_t_plot(A, lat, lon, name=place, fout=fout)

      # Meteogram
      day_nc = A.paths["data_meteograms"] / f"meteogram_{code}.nc"
      ds = make_meteogram_timestep(A, lat, lon)
      ds_full = append_to_meteogram(ds, day_nc)
      if len(ds_full["time"]) >= 2:
         fout = A.paths["plots_daily"] / f"meteogram_{code}.png"
         plot_meteogram(day_nc, fout=fout)
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

   LG.info("==== New run started ====")
   try:
      process_file(fname, config_file, LG)
   except Exception as e:
      LG.exception(f"Failed to process file {fname}: {e}")
      sys.exit(1)

if __name__ == "__main__":
   main()































###   exit()
###   import utils as ut
###   import os
###   here = os.path.dirname(os.path.realpath(__file__))
###   is_cron = bool( os.getenv('RUN_BY_CRON') )
###   
###   import sys
###   try: fname = sys.argv[1]
###   except IndexError:
###      print('File not specified\nUsage: pyhton <this script> /path/to/ncfile')
###      sys.exit(1)
###   
###   folder = '../../Documents/storage/WRFOUT/Spain6_1'
###   date = '2025-05-10'
###   fname = f'{folder}/wrfout_d02_{date}_07:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_08:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_09:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_10:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_11:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_12:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_13:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_14:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_15:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_16:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_17:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_18:00:00'
###   # fname = f'{folder}/wrfout_d02_{date}_19:00:00'
###   
###   
###   
###   
###   from calc_data import CalcData
###   
###   
###   
###   
###   from time import time
###   
###   
###   output_folder,plots_folder,data_folder = ut.get_folders()
###   told = time()
###   A = CalcData(fname, OUT_folder=plots_folder, DATA_folder=data_folder)
###   print(f'Process WRF: {time()-told:.5f}s')
###   
###   
###   
###   lat, lon = 41.1, -3.6
###   # # index
###   # i,j = 108, 192
###   
###   # Plot soundings & meteograms
###   import pandas as pd
###   from plots.sounding import skew_t_plot
###   from meteogram_writer import make_meteogram_timestep, append_to_meteogram
###   from plots.meteogram import plot_meteogram
###   df = pd.read_csv(f'soundings_{domain}.csv', 
###                     names=['lat', 'lon', 'code', 'place'])
###   for _,row in df.iterrows():
###      lat,lon,code,place = row['lat'], row['lon'], row['code'], row['place']
###      # Soundigs
###      fout = f'{A.OUT_folder}/{A.tail_h}_sounding_{code}.png'
###      skew_t_plot(A, lat,lon,name=place,fout=fout)
###      # Meteogram
###      day_nc = f"meteogram_{code}_{A.tail_d}.nc"  # stores the data for that day
###      ds = make_meteogram_timestep(A, lat, lon)
###      ds_full = append_to_meteogram(ds, day_nc)
###      if len(ds_full["time"]) >= 2:
###         fout = f'{A.OUT_folder}/meteogram_{code}.png'
###         plot_meteogram(day_nc, fout=fout)
###   
###   
###   # 
###   # # TODO
###   # from plots.web import generate_background, generate_scalar_fields,
###   # from plots.web import generate_vector_fields
###   # 
###   # # Background (just once per day, or reused across plots)
###   # generate_background(domain=A.domain, date=A.date, out_dir=plots_folder)
###   # 
###   # # Scalars: T2, RH, Wstar, CAPE, etc
###   # generate_scalar_fields(A, out_dir=plots_folder)
###   # 
###   # # Vectors: winds, streamlines, etc
###   # generate_vector_fields(A, out_dir=plots_folder)
###   # 
###   
###   
###   
###   exit()
###   # # output_folder = expanduser( P['system']['output_folder'] )
###   # # plots_folder = expanduser( P['system']['plots_folder'] )
###   # # data_folder = expanduser( P['system']['data_folder'] )
###   # ut.check_directory(output_folder,True)
###   # ut.check_directory(output_folder+'/processed',False)
###   # ut.check_directory(plots_folder,False)
###   # ut.check_directory(data_folder,False)
###               
###   zooms = ut.get_zooms('zooms.ini',domain=domain)
###   
###   import numpy as np
###   print(np.min(hcrit:=A.drjack_vars['hcrit'].values))
###   print(np.max(hcrit))
###   
###   import matplotlib.pyplot as plt
###   try: plt.style.use('mystyle')
###   except: pass
###   fig, ax = plt.subplots()
###   ax.contourf(hcrit, vmin=300, vmax=2500)
###   fig.tight_layout()
###   plt.show()
###   
