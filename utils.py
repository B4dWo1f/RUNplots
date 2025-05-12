#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from configparser import ConfigParser, ExtendedInterpolation
from os.path import expanduser
import datetime as dt
fmt =  '%Y-%m-%d_%H:%M:%S'

def get_domain(fname):
   return fname.split('/')[-1].replace('wrfout_','').split('_')[0]

def file2date(fname):
   date = fname.split('/')[-1]
   date = '_'.join(date.split('_')[-2:])
   return dt.datetime.strptime(date,fmt)

def date2file(date,domain,folder):
   return f'{folder}/wrfout_{domain}_{date.strftime(fmt)}'

def get_config(fname='plots.ini'):
   """
   Return the data for plotting property. Intended to read from plots.ini
   """
   # LG.info(f'Loading config file: {fname}')
   # if not os.path.isfile(fname): return None
   config = ConfigParser(inline_comment_prefixes='#')
   config._interpolation = ExtendedInterpolation()
   config.read(fname)
   props = config._sections
   for item,value in props.items():
      for k,v in value.items():
         if k in ['factor','delta','vmin','vmax']: value[k] = float(v)
   return props

def get_folders(fname='plots.ini'):
   # LG.info(f'Loading config file: {fname}')
   # if not os.path.isfile(fname): return None
   config = ConfigParser(inline_comment_prefixes='#')
   config._interpolation = ExtendedInterpolation()
   config.read(fname)
   return expanduser(config['system']['output_folder']),\
          expanduser(config['system']['plots_folder']),\
          expanduser(config['system']['data_folder'])

def get_zooms(fname='zooms.ini',domain=''):
   sects = get_config(fname)
   zooms = []
   for k,v in sects.items():
      parent = v['parent']
      left   = float(v['left'])
      right  = float(v['right'])
      bottom = float(v['bottom'])
      top    = float(v['top'])
      if len(domain) > 0:
         if parent == domain:
            zooms.append( [left,right,bottom,top] )
         else: pass
      else: zooms.append( [left,right,bottom,top] )
   return zooms


#################################################################################
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
