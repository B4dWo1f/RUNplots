#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import logging
LG = logging.getLogger(f'main.{__name__}')
LGp = logging.getLogger(f'perform.{__name__}')

import os
here = os.path.dirname(os.path.realpath(__file__))
HOME = os.getenv('HOME')
from configparser import ConfigParser, ExtendedInterpolation
from os.path import expanduser
from pathlib import Path
import sys
import datetime as dt
fmt =  '%Y-%m-%d_%H:%M:%S'

# from utils import check_directory  # your existing util


def check_directory(path, create=True):
   """
   Check if a folder exists, and optionally create it if it does not.
   ___ Parameters ___
   path: [str] The directory path to check
   create: [bool] Whether to create the directory if it does not exist
   ___ Returns ___
   bool: True if the directory exists or was created successfully, False
         otherwise
   """
   path = str(path)  # ensure compatibility with Path objects
   if os.path.isdir(path):
      LG.debug(f'Folder {path} already existed')
      return True
   if create:
      try:
         LG.debug(f'Creating folder: {path}')
         os.makedirs(path, exist_ok=True)
         LG.info(f'Created folder: {path}')
         return True
      except OSError as e:
         LG.critical(f"Failed to create directory '{path}': {e}")
         return False
   return False

def get_GFSbatch(path):
   """Read the GFS batch. It is expected to be in {wrfout_folder}/batch.txt """
   fmt = '%d/%m/%Y-%H:%M'
   try:
      with open(path,'r') as f:
         gfs_batch = f.read().strip()
         gfs_batch = dt.datetime.strptime(gfs_batch, fmt)
   except:
      LG.critical(f'Unable to determine GFS batch from {path}')
      gfs_batch = '???'
   return gfs_batch

def get_domain(fname):
   # return fname.split('/')[-1].replace('wrfout_','').split('_')[0]
   return fname.name.replace('wrfout_', '').split('_')[0]

def file2date(fname):
   date = fname.name.split('_')[-2:]
   date = '_'.join(date)
   return dt.datetime.strptime(date,fmt)

def date2file(date,domain,folder):
   return f'{folder}/wrfout_{domain}_{date.strftime(fmt)}'


REQUIRED_SECTIONS = {
    "paths": ["wrfout_folder", "plots_folder", "data_folder"]
}

def load_config_or_die(fname="{here}/config.ini", create_dirs=True):
   """
   Load and validate config file. Fails loudly if required fields are missing.
   
   Returns:
       config (ConfigParser)
       paths (dict[str, Path])
   """
   config_path = Path(fname).expanduser().resolve()
   base_dir = config_path.parent  # For resolving relative paths

   config = ConfigParser(interpolation=ExtendedInterpolation())
   config.read(fname)

   # Check required keys
   for section, keys in REQUIRED_SECTIONS.items():
      if section not in config:
         LG.critical(f"[ERROR] Missing section: [{section}] in {fname}")
         sys.exit(1)
      for key in keys:
         if key not in config[section]:
            msg = f"[ERROR] Missing '{key}' in section [{section}] of {fname}"
            LG.critical(msg)
            sys.exit(1)

   #
   # Normally
   #   domain = d02
   #   wrfout_folder = /storage/WRFOUT/Spain6_1
   #   plots_folder  = /storage/PLOTS/Spain6_1/<domain>
   #   data_folder   = /storage/DATA/Spain6_1/<domain>
   #
   domain = config["paths"]["domain"]
   wrfout_folder  = Path(expanduser(config["paths"]["wrfout_folder"]))
   plots_folder   = Path(expanduser(config["paths"]["plots_folder"]))
   data_folder    = Path(expanduser(config["paths"]["data_folder"]))

   raw_configs_path = config["paths"]["configs"]
   configs_folder = (base_dir / raw_configs_path).expanduser().resolve()
   # configs_folder = Path(expanduser(config["paths"]["configs"]))
   mydict = {
         'wrfout_folder'  : wrfout_folder,
         "plots_folder"   : plots_folder,
         "data_folder"    : data_folder,
         #
         "configs_folder" : configs_folder,
         "data_stations"  : data_folder/'stations', #domain/date_fmt,
         "plots_stations" : plots_folder/'stations',
         }

   for label,path in mydict.items():
      check_directory(path)
   plots_path = configs_folder / 'plots.ini'
   zooms_path = configs_folder / 'zooms.ini'
   for path in [plots_path, zooms_path]:
      if not path.exists():
         LG.critical(f"Mandatory file {path} is not present")
         sys.exit(1)
   mydict['plots_ini'] = plots_path
   mydict['zooms_ini'] = zooms_path
   return mydict


def load_zooms(zoom_file, domain=None):
    """
    Load zoom definitions from an INI file and filter by WRF domain.

    Parameters
    ----------
    zoom_file : [str or Path] Path to the zooms.ini file.
    domain : [str] optional. If specified, only return zooms whose 'parent'
                   matches the domain (e.g. 'd02').
    Returns
    -------
    dict: Dictionary of zoom_name -> (left, right, bottom, top)
    """
    config = ConfigParser()
    config.read(zoom_file)

    zooms = {}
    for section in config.sections():
        parent = config[section].get("parent", "").strip()
        if (domain is None) or (parent == domain):
            bounds = tuple(map(float, [
                config[section]["left"],
                config[section]["right"],
                config[section]["bottom"],
                config[section]["top"]
            ]))
            zooms[section] = bounds

    return zooms






###############################################################################
def pretty_print_var(data):
   """
   Pretty-print summary for a WRF xarray.DataArray variable.
   """
   summary = {"Description": data.attrs.get("description", "N/A"),
              "Name": data.name,
              "Units": data.attrs.get("units", "N/A"),
              "Shape": data.shape,
              "Dimensions": data.dims,
              "No coord dims": list(set(data.dims) - set(data.coords)),
              "Dtype": str(data.dtype),
              "Fill value": data.attrs.get("_FillValue", "N/A"),
              "Time": str(data.coords.get("Time", "N/A").values) if "Time" in data.coords else "N/A"
   }

   separator =  "─" * 63 + '\n'
   spacing = ' ' * ((len(separator) - len(data.name)-2)//2)
   msg =  separator
   msg += f"{spacing} {data.name}\n"
   # msg += f"{'Field':<13} | {'Value'}\n"
   msg += separator
   for key, value in summary.items():
      msg += f" {key:<13} | {value}\n"
   msg += separator
   return msg
