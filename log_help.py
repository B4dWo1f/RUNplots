#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# import logging
# LG = logging.getLogger(__name__)   # Logger for this module
#LlG = logging.getLogger('perform')
#fh = logging.FileHandler('/tmp/performance.log',mode='w')
#fmt = logging.Formatter('%(asctime)s %(name)s:%(levelname)s - %(message)s')
#fh.setFormatter(fmt)
#fh.setLevel(logging.DEBUG)
#LlG.addHandler(fh)

import os
import logging
import datetime as dt
import utils as ut
from pathlib import Path
from functools import wraps
from time import perf_counter

# def get_current_batch(path="/storage/WRFOUT/batch.txt"):
#     try:
#         with open(path, 'r') as f:
#             batch = f.read().strip()
#             return batch
#     except Exception:
#         return "UNKNOWN"

def batch_logger(script_path, domain, batch, is_cron=False, log_dir=None):
   """
   Sets up a logger that writes to a log file named according to the current
   date and batch
   Optionally adds screen logging if not run by cron
   """
   log_dir = Path(log_dir)
   if not log_dir.is_absolute():
      # Resolve relative to the file where this function lives
      log_dir = Path(__file__).resolve().parent / log_dir
   script_path = Path(script_path).resolve()
   base_name = script_path.stem
   log_name = f"{base_name}_{domain}_GFS{batch.strftime('%H')}"
   # base = os.path.splitext(os.path.basename(script_path))[0]
   # log_name = f"{base}_{domain}_GFS{batch.strftime('%H')}.log"

   # log_dir = log_dir or os.path.dirname(script_path)
   # log_path = os.path.join(log_dir, log_name)
   log_dir = Path(log_dir) if log_dir else script_path.parent
   ut.check_directory(log_dir)

   log_path  = log_dir / f"{log_name}.log"
   perf_path = log_dir / f"{log_name}.perform"

   # Create and configure loggers
   logger = logging.getLogger('main')
   logger.setLevel(logging.DEBUG)

   perform = logging.getLogger('perform')
   perform.setLevel(logging.DEBUG)

   formatter = logging.Formatter(
       '%(asctime)s %(name)s:%(levelname)s - %(message)s',
       datefmt='%Y/%m/%d-%H:%M:%S'
   )

   # Main logger file handler
   file_handler = logging.FileHandler(log_path, mode='a')
   file_handler.setFormatter(formatter)
   logger.addHandler(file_handler)

   # Performance logger file handler
   perf_handler = logging.FileHandler(perf_path, mode='a')
   perf_handler.setFormatter(formatter)
   perform.addHandler(perf_handler)

   if not is_cron:
       screen_handler(logger, lv=logging.DEBUG)

   msg = f"Started script: {script_path} | domain: {domain} | batch: {batch}"
   logger.info(msg)
   return logger, perform



def screen_handler(lg=None,lv='debug',fmt='%(name)s -%(levelname)s- %(message)s'):
   """
     This function adds a screenHandler to a given logger. If no logger is
     specified, just return the handler
   """
   if lv == 'debug': lv = logging.DEBUG
   elif lv == 'info': lv = logging.INFO
   elif lv == 'warning': lv = logging.WARNING
   elif lv == 'error': lv = logging.ERROR
   elif lv == 'critical': lv = logging.CRITICAL
   sh = logging.StreamHandler()
   sh.setLevel(lv)
   fmt = logging.Formatter(fmt)
   sh.setFormatter(fmt)
   if lg != None: lg.addHandler(sh)
   return sh

## Timer Decorator
def timer(lg_progression,lg_performance):
   """Logs the execution time of a certain function to the provided logger"""
   def real_timer(wrapped):
      @wraps(wrapped)   # preserve metadata about the function being decorated
      def inner(*args, **kwargs):
         t0 = perf_counter()
         lg_progression.debug(f'Entering {wrapped.__name__}')
         ret = wrapped(*args, **kwargs)
         lg_progression.debug(f'Finished {wrapped.__name__}')
         t1 = perf_counter()
         lg_performance.debug(f'Time for {wrapped.__name__}: {t1-t0:.4f}s')
         return ret
      return inner
   return real_timer


def log2screen(lg,lv='info'):
   """ This decorator adds *temporarily* a screenHandler to a given logger """
   def do_it(wrapped):
      def inner(*args, **kwargs):
         sh = screen_handler(lg,lv=lv)  # add ScreenHandler
         ret = wrapped(*args, **kwargs)
         lg.removeHandler(sh)     # remove ScreenHandler
         return ret
      return inner
   return do_it


def disable(lg):
   """Temporarily raise the log level to CRITICAL to avoid over logging"""
   def do_it(wrapped):
      def inner(*args, **kwargs):
         lv = lg.getEffectiveLevel()
         lg.setLevel(logging.CRITICAL)
         ret = wrapped(*args, **kwargs)
         lg.setLevel(lv)
         return ret
      return inner
   return do_it

def disable2(lg):
   """Temporarily raise the log level to INFO to avoid over logging"""
   def do_it(wrapped):
      def inner(*args, **kwargs):
         lv = lg.getEffectiveLevel()
         lg.setLevel(logging.INFO)
         ret = wrapped(*args, **kwargs)
         lg.setLevel(lv)
         return ret
      return inner
   return do_it



# def inout(lg):
#    """Logs the execution time of a certain funcion to the provided logger"""
#    def real_timer(wrapped):
#       def inner(*args, **kwargs):
#          lg.debug(f'Entering {wrapped.__name__}')
#          ret = wrapped(*args, **kwargs)
#          lg.debug(f'Finished {wrapped.__name__}')
#          return ret
#       return inner
#    return real_timer


# def deprecated(lg):
#    """Issues a warning when using a deprecated function"""
#    def wrapper(wrapped):
#       def inner(*args, **kwargs):
#          lg.warning(f'Deprecated use of {wrapped.__name__}')
#          ret = wrapped(*args, **kwargs)
#          return ret
#       return inner
#    return wrapper

