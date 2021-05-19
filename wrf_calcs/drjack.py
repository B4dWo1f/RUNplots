#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from . import util as ut
import log_help
import logging
LG = logging.getLogger(__name__)
import os
here = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from . import extract

def calculations(ncfile,wa,heights,terrain,pressure,p,pb,bldepth,hfx,qvapor,qcloud,tc,td,my_cache=None):
   ## Derived Quantities by DrJack ##############################################
   # Using utils wrappers to hide the transpose of every variable
   # XXX Probalby Inefficient
   # BL Max. Up/Down Motion (BL Convergence)______________________[cm/s] (ny,nx)
   wblmaxmin = ut.calc_wblmaxmin(0, wa, heights, terrain, bldepth)
   LG.debug(f'wblmaxmin: {wblmaxmin.shape}')
 
   # Thermal Updraft Velocity (W*)_________________________________[m/s] (ny,nx)
   wstar = ut.calc_wstar( hfx, bldepth )
   LG.debug(f'W*: {wstar.shape}')
 
   # BLcwbase________________________________________________________[m] (ny,nx)
   # laglcwbase = 0 --> height above sea level
   # laglcwbase = 1 --> height above ground level
   laglcwbase = 0
   # criteriondegc = 1.0
   maxcwbasem = 5486.40
   cwbasecriteria = 0.000010
   blcwbase = ut.calc_blcloudbase( qcloud,  heights, terrain, bldepth,
                                   cwbasecriteria, maxcwbasem, laglcwbase)
   LG.debug(f'blcwbase: {blcwbase.shape}')
 
   # Height of Critical Updraft Strength (hcrit)_____________________[m] (ny,nx)
   hcrit = ut.calc_hcrit( wstar, terrain, bldepth)
   LG.debug(f'hcrit: {hcrit.shape}')
 
   # Height of SFC.LCL_______________________________________________[m] (ny,nx)
   # Cu Cloudbase ~I~where Cu Potential > 0~P~
   zsfclcl = ut.calc_sfclclheight( pressure, tc, td, heights, terrain, bldepth )
   LG.debug(f'zsfclcl: {zsfclcl.shape}')
 
   # OvercastDevelopment Cloudbase_______________________________[m?] (nz,ny,nx)
   pmb = 0.01*(p.values+pb.values) # press is vertical coordinate in mb
   zblcl = ut.calc_blclheight(qvapor,heights,terrain,bldepth,pmb,tc)
   LG.debug(f'zblcl: {zblcl.shape}')
 
   # Thermalling Height______________________________________________[m] (ny,nx)
   hglider = np.minimum(np.minimum(hcrit,zsfclcl), zblcl)
   LG.debug(f'hglider: {hglider.shape}')
 
   # Mask zsfclcl, zblcl________________________________________________________
   ## Mask Cu Pot > 0
   zsfclcldif = bldepth + terrain - zsfclcl
   null = 0. * zsfclcl
   # cu_base_pote = np.where(zsfclcldif>0, zsfclcl, null)
   zsfclcl = np.where(zsfclcldif>0, zsfclcl, null)
   LG.debug(f'zsfclcl mask: {zsfclcl.shape}')
 
   ## Mask Overcast dev Pot > 0
   zblcldif = bldepth + terrain - zblcl
   null = 0. * zblcl
   # over_base_pote = np.where(zblcldif>0, zblcl, null)
   zblcl = np.where(zblcldif>0, zblcl, null)
   LG.debug(f'zblcl mask: {zblcl.shape}')

   # BL Avg Wind__________________________________________________[m/s?] (ny,nx)
   # uv NOT rotated to grid in m/s
   uv = extract.getvar(ncfile, "uvmet", cache=my_cache)
   uEW = uv[0,:,:,:]
   vNS = uv[1,:,:,:]
   ublavgwind = ut.calc_blavg(uEW, heights, terrain, bldepth)
   vblavgwind = ut.calc_blavg(vNS, heights, terrain, bldepth)
   LG.debug(f'uBLavg: {ublavgwind.shape}')
   LG.debug(f'vBLavg: {vblavgwind.shape}')

   # BL Top Wind__________________________________________________[m/s?] (ny,nx)
   utop,vtop = ut.calc_bltopwind(uEW, vNS, heights,terrain,bldepth)
   LG.debug(f'utop: {utop.shape}')
   LG.debug(f'vtop: {vtop.shape}')
   return wblmaxmin, wstar, blcwbase, hcrit, zsfclcl, zblcl, hglider, ublavgwind, vblavgwind, utop,vtop

