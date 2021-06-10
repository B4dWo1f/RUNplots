#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
here = os.path.dirname(os.path.realpath(__file__))
import log_help
import logging
LG = logging.getLogger(__name__)
LG.setLevel(logging.DEBUG)

## True unless RUN_BY_CRON is not defined
is_cron = bool( os.getenv('RUN_BY_CRON') )
import matplotlib as mpl
# if is_cron:
#    LG.warning('Run from cron. Using Agg backend')
mpl.use('Agg')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 15.0
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LightSource, BoundaryNorm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import colormaps as mcmaps
from scipy.interpolate import interp1d
from matplotlib.patches import Circle

# Map
import numpy as np
import rasterio
from rasterio.merge import merge
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature

# Sounding
from metpy.plots import SkewT, Hodograph
from metpy.units import units
import metpy.calc as mpcalc


## Dark Theme ##################################################################
#  COLOR = 'black'
#  ROLOC = '#e0e0e0'
# mpl.rcParams['axes.facecolor'] = (1,1,1,0)
# mpl.rcParams['figure.facecolor'] = (1,1,1,0)
# mpl.rcParams["savefig.facecolor"] = (1,1,1,0)
#  mpl.rcParams['text.color'] = ROLOC #COLOR
#  mpl.rcParams['axes.labelcolor'] = COLOR
#  mpl.rcParams['axes.facecolor'] = COLOR #'black'
#  mpl.rcParams['savefig.facecolor'] = COLOR #'black'
#  mpl.rcParams['xtick.color'] = COLOR
#  mpl.rcParams['ytick.color'] = COLOR
#  mpl.rcParams['axes.edgecolor'] = COLOR
#  mpl.rcParams['axes.titlesize'] = 20
################################################################################

mycolormaps = {'WindSpeed': mcmaps.WindSpeed,
               'Convergencias': mcmaps.Convergencias,
               'CAPE': mcmaps.CAPE, 'Rain': mcmaps.Rain, 'None': None,
               'greys': mcmaps.greys, 'reds': mcmaps.reds,
               'greens': mcmaps.greens, 'blues': mcmaps.blues}

@log_help.timer(LG)
def setup_plot(ref_lat,ref_lon,left,right,bottom,top,transparent=True):
   mpl.rcParams['axes.facecolor'] = (1,1,1,0)
   mpl.rcParams['figure.facecolor'] = (1,1,1,0)
   mpl.rcParams["savefig.facecolor"] = (1,1,1,0)
   orto = ccrs.PlateCarree()
   projection = ccrs.LambertConformal(ref_lon,ref_lat)
   # projection = ccrs.RotatedPole(pole_longitude=-150, pole_latitude=37.5)

   extent = left, right, bottom, top
   fig = plt.figure(figsize=(11,9)) #, frameon=False)
   # ax = plt.axes(projection=projection)
   ax = fig.add_axes([0,0,0.99,1],projection=projection)
   ax.set_extent(extent, crs=orto)
   if transparent:
       # ax.outline_patch.set_visible(False)
       ax.background_patch.set_visible(False)
   return fig,ax,orto

def save_figure(fig,fname,dpi=150, quality=90):
   LG.info(f'Saving: {fname}')
   fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0,
                      dpi=dpi, quality=quality)
   plt.close('all')

@log_help.timer(LG)
def terrain(reflat,reflon,left,right,bottom,top,ve=0.3):
   fig, ax, orto = setup_plot(reflat,reflon,left,right,bottom,top)
### RASTER ###################################################################
   files = os.popen('ls terrain_tif/geb*').read().strip().splitlines()
   srcs = [rasterio.open(fname, 'r') for fname in files]
   # fname0 = files[0]
   # fname1 = files[1]
   # src0 = rasterio.open(fname0, 'r')
   # src1 = rasterio.open(fname1, 'r')
   D = 2
   mosaic, out_trans = merge(srcs, (left-D, bottom-D, right+D, top+D))
   terrain = mosaic[0,:,:]
   ls = LightSource(azdeg=315, altdeg=65)
   terrain = ls.hillshade(terrain, vert_exag=ve)
   # from scipy.ndimage.filters import gaussian_filter
   # ax.imshow(gaussian_filter(terrain,1),
   ax.imshow(terrain, extent=(left-D, right+D, bottom-D, top+D),
                      origin='upper', cmap='gray',
                      aspect='equal', interpolation='bicubic',
                      zorder=0, transform=orto)
   return fig,ax,orto

@log_help.timer(LG)
def parallel_and_meridian(fig,ax,orto,left,right,bottom,top,nx=1,ny=1):
   lcs = 'k--'
   D = 1
   # Plotting meridian
   for x in range(int(left-D), int(right+D)):
      if x%nx ==0:
         ax.plot([x,x],[bottom-D,top+D], lcs, transform=orto)
   # Plotting parallels
   for y in range(int(bottom-D), int(top+D)):
      if y%ny == 0:
         ax.plot([left-D,right+D],[y,y], lcs, transform=orto)
   return fig,ax,orto

@log_help.timer(LG)
def rivers_plot(fig,ax,orto):
   rivers = NaturalEarthFeature('physical',
                                'rivers_lake_centerlines_scale_rank',
                                '10m', facecolor='none')
   ax.add_feature(rivers, lw=2 ,edgecolor='C0',zorder=50)
   rivers = NaturalEarthFeature('physical', 'rivers_europe',
                                '10m', facecolor='none')
   ax.add_feature(rivers, lw=2 ,edgecolor='C0',zorder=50)
   for field in ['lakes','lakes_historic','lakes_pluvial','lakes_europe']:
       water = NaturalEarthFeature('physical', field, '10m')
       ax.add_feature(water, lw=2 ,edgecolor='C0',
                      facecolor=cfeature.COLORS['water'],zorder=50)
   return fig,ax,orto

@log_help.timer(LG)
def sea_plot(fig,ax,orto):
   """
   XXX Not working
   """
   sea = NaturalEarthFeature('physical', 'bathymetry_all', '10m') #, facecolor='none')
   ax.add_feature(sea, lw=2) # ,edgecolor='C0',zorder=50)
   return fig,ax,orto

@log_help.timer(LG)
def ccaa_plot(fig,ax,orto):
   provin = NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines',
                                            '10m', facecolor='none')
   country = NaturalEarthFeature('cultural', 'admin_0_countries', '10m',
                                                   facecolor='none')
   ax.add_feature(provin, lw=2 ,edgecolor='k',zorder=50)
   ax.add_feature(country,lw=2.3, edgecolor='k', zorder=51)
   return fig,ax,orto

@log_help.timer(LG)
def road_plot(fig,ax,orto):
   roads = NaturalEarthFeature('cultural', 'roads',
                                            '10m', facecolor='none')
   ax.add_feature(roads, lw=2 ,edgecolor='w',zorder=51)
   ax.add_feature(roads, lw=3 ,edgecolor='k',zorder=50)
   return fig,ax,orto

@log_help.timer(LG)
def csv_plot(fig,ax,orto, fname,marker='x'):
   Yt,Xt = np.loadtxt(fname,usecols=(0,1),delimiter=',',unpack=True)
   names = np.loadtxt(fname,usecols=(2,),delimiter=',',dtype=str)
   ax.scatter(Xt,Yt,s=40,c='r',marker=marker,transform=orto,zorder=53)
   return fig,ax,orto

@log_help.timer(LG)
def csv_names_plot(fig,ax,orto, fname):
   # Cities
   Yt,Xt = np.loadtxt(fname,usecols=(0,1),delimiter=',',unpack=True)
   names = np.loadtxt(fname,usecols=(2,),delimiter=',',dtype=str)
   for x,y,name in zip(Xt,Yt,names):
      txt = ax.text(x,y,name, horizontalalignment='center',
                              verticalalignment='center',
                              color='k',fontsize=13,
                              transform=orto,zorder=52)
      txt.set_path_effects([PathEffects.withStroke(linewidth=5,
                                                   foreground='w')])
      txt.set_clip_on(True)
   return fig,ax,orto


@log_help.timer(LG)
def scalar_plot(fig,ax,orto, lons,lats,prop, delta,vmin,vmax,cmap,
                                                 levels=[], inset_label=''):
   """
   Plot a scalar property. 
   fig: matplotlib figure to plot in. XXX unnecessary??
   ax: axis to plot in
   orto: geographical projection (transform argument)
   lons,lats: (nx,ny) matrix of grid longitudes and latitudes
   prop: (nx,ny) matrix of the scalar property
   vmin,vmax,delta: min, max, and step for colormap
   cmap: string of colormap it has to be defined in the local dictionary mycolormaps or be an acceptable matplotlib.colormap name
   Several options for the levels and color range:
   levels = None: levels remain none and are, hence, ignored
   levels = []  : a list of levels is computed from vmin to vmax with delta steps
   levels = [#] : the list provided is respected
   inset_label: text to appear in the lower right corner of the graph
   """
   if type(levels) == type(None):
      norm = None
   else:
      if len(levels) > 0:
         norm = BoundaryNorm(levels,len(levels))
      else:
         levels = np.arange(vmin,vmax,delta)
         norm = None
   try: cmap = mycolormaps[cmap]
   except KeyError: pass
   try:
      C = ax.contourf(lons,lats,prop, levels=levels, extend='max',
                                      antialiased=True, norm=norm,
                                      cmap=cmap, vmin=vmin, vmax=vmax,
                                      zorder=10, transform=orto)
   except:
       # LG.warning('NaN values found, unable to plot')
       C = None
   if len(inset_label) > 0:
      txt = ax.text(1,0, inset_label, va='bottom', ha='right', color='k',
                    fontsize=12, bbox=dict(boxstyle="round",
                                           ec=None, fc=(1,1,1,.9)),
                    zorder=100, transform=ax.transAxes)
      txt.set_clip_on(True)
   return C

@log_help.timer(LG)
def vector_plot(fig,ax,orto,lons,lats,U,V, dens=1.5,color=(0,0,0,0.75)):
   """
   Plot a vector property. 
   fig: matplotlib figure to plot in. XXX unnecessary??
   ax: axis to plot in
   orto: geographical projection (transform argument)
   lons,lats: (nx,ny) matrix of grid longitudes and latitudes
   U,V: (nx,ny) U and V components of the vector field
   dens: density of arrows in the map
   color: color of the arrows
   """
   x = lons #[0,:]
   y = lats #[:,0]
   ax.streamplot(x,y, U,V, color=color, linewidth=1, density=dens,
                           arrowstyle='->',arrowsize=2.5,
                           zorder=11,
                           transform=orto)

@log_help.timer(LG)
def barbs_plot(fig,ax,orto,lons,lats,U,V, n=1,color=(0,0,0,0.75)):
   """
   Plot a wind barbs
   fig: matplotlib figure to plot in. XXX unnecessary??
   ax: axis to plot in
   orto: geographical projection (transform argument)
   lons,lats: (nx,ny) matrix of grid longitudes and latitudes
   U,V: (nx,ny) U and V components of the vector field
   """
   n = 1
   f = 2
   ax.barbs(lons[::n,::n],lats[::n,::n], U[::n,::n],V[::n,::n],
            color=color, length=4, pivot='middle',
            sizes=dict(emptybarb=0.25/f, spacing=0.2/f, height=0.5/f),
            linewidth=0.75, transform=orto)



@log_help.timer(LG)
def plot_colorbar(cmap,delta=4,vmin=0,vmax=60,levels=None,name='cbar',
                                        units='',fs=18,norm=None,extend='max'):
   """
   Generate colorbar for the colormap cmap.
   vmin,vmax,delta: min, max, and step for colormap
   cmap: string of colormap it has to be defined in the local dictionary mycolormaps or be an acceptable matplotlib.colormap name
   Several options for the levels and color range:
   levels = None: levels remain none and are, hence, ignored
   levels = []  : a list of levels is computed from vmin to vmax with delta steps
   levels = [#] : the list provided is respected
   """
   try: cmap = mycolormaps[cmap]
   except KeyError: pass
   fig, ax = plt.subplots()
   fig.set_figwidth(11)
   img = np.random.uniform(vmin,vmax,size=(40,40))
   if type(levels) != type(None) and len(levels) == 0:
      levels=np.arange(vmin,vmax,delta)
   img = ax.contourf(img, levels=levels,
                          extend=extend,
                          antialiased=True,
                          cmap=cmap,
                          norm=norm,
                          vmin=vmin, vmax=vmax)
   plt.gca().set_visible(False)
   divider = make_axes_locatable(ax)
   cax = divider.new_vertical(size="5%", pad=0.25, pack_start=True)
           #2.95%"
   fig.add_axes(cax)
   cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
   cbar.ax.set_xlabel(units,fontsize=fs)
   fig.savefig(f'{name}.png', transparent=True,
                              bbox_inches='tight', pad_inches=0.1)
   plt.close('all')  #XXX are you sure???


def manga(fig,ax,orto):
   f_manga = f'{here}/task.gps'
   ang = np.arctan(1/6371)
   ang = 1/111
   try:
      y,x,Rm = np.loadtxt(f_manga,usecols=(0,1,2),delimiter=',',unpack=True)
      cont = 0
      for ix,iy,r in zip(x,y,Rm):
         if cont == 0: color = 'C1'
         elif cont == len(y)-2: color = 'C3'
         elif cont == len(y)-1: color = 'C0'
         else: color = 'C2'
         ax.add_patch(Circle(xy=[ix,iy], radius=(r/1000)*ang,
                             color=color, alpha=0.3, transform=orto, zorder=30))
         cont += 1

      # spacing of arrows
      scale = 2
      aspace = .18 # good value for scale of 1
      aspace *= scale

      # r is the distance spanned between pairs of points
      r = [0]
      for i in range(1,len(x)):
          dx = x[i]-x[i-1]
          dy = y[i]-y[i-1]
          r.append(np.sqrt(dx*dx+dy*dy))
      r = np.array(r)

      # rtot is a cumulative sum of r, it's used to save time
      rtot = []
      for i in range(len(r)):
          rtot.append(r[0:i].sum())
      rtot.append(r.sum())

      arrowData = [] # will hold tuples of x,y,theta for each arrow
      arrowPos = 0   # current point on walk along data
      rcount = 1
      while arrowPos < r.sum():
          x1,x2 = x[rcount-1],x[rcount]
          y1,y2 = y[rcount-1],y[rcount]
          da = arrowPos-rtot[rcount]
          theta = np.arctan2((x2-x1),(y2-y1))
          X = np.sin(theta)*da+x1
          Y = np.cos(theta)*da+y1
          arrowData.append((X,Y,theta))
          arrowPos+=aspace
          while arrowPos > rtot[rcount+1]:
              rcount+=1
              if arrowPos > rtot[-1]: break

      # could be done in above block if you want
      for X,Y,theta in arrowData:
          # use aspace as a guide for size and length of things
          # scaling factors were chosen by experimenting a bit
          ax.arrow(X,Y,
                     np.sin(theta)*aspace/10,np.cos(theta)*aspace/10,
                     head_width=aspace/3, color='C3', transform=orto)
      # ax.plot(x,y)
      ax.plot(x,y, 'C3-', lw=2, transform=orto) #c='C4',s=50,zorder=20)
   except: pass




# Sounding
## Dark Theme ##################################################################
# #  COLOR = 'black'
# #  ROLOC = '#e0e0e0'
#  mpl.rcParams['text.color'] = ROLOC #COLOR
#  mpl.rcParams['axes.labelcolor'] = COLOR
#  mpl.rcParams['axes.facecolor'] = COLOR #'black'
#  mpl.rcParams['savefig.facecolor'] = COLOR #'black'
#  mpl.rcParams['xtick.color'] = COLOR
#  mpl.rcParams['ytick.color'] = COLOR
#  mpl.rcParams['axes.edgecolor'] = COLOR
#  mpl.rcParams['axes.titlesize'] = 20
################################################################################

def find_cross(left,right,p,tc,interp=True):
   if interp:
      ps = np.linspace(np.max(p),np.min(p),500)
      left = interp1d(p,left)(ps)
      right = interp1d(p,right)(ps)
      aux = (np.diff(np.sign(left-right)) != 0)*1
      ind, = np.where(aux==1)
      ind_cross = np.min(ind)
      # ind_cross = np.argmin(np.abs(left-right))
      p_base = ps[ind_cross]
      t_base = right[ind_cross]
   else:
      aux = (np.diff(np.sign(left-right)) != 0)*1
      ind, = np.where(aux==1)
      ind_cross = np.min(ind)
      p_base = p[ind_cross]
      t_base = tc[ind_cross]
   return p_base, t_base

#@log_help.timer(LG)
#def skewt_plot(p,tc,tdc,t0,date,u=None,v=None,fout='sounding.png',latlon='',title='',show=False):
#   """
#   h: heights
#   p: pressure
#   tc: Temperature [C]
#   tdc: Dew point [C]
#   date: date of the forecast
#   u,v: u,v wind components
#   adapted from:
#   https://geocat-examples.readthedocs.io/en/latest/gallery/Skew-T/NCL_skewt_3_2.html#sphx-glr-gallery-skew-t-ncl-skewt-3-2-py
#   """
#   mpl.rcParams['axes.facecolor'] = (1,1,1,1)
#   mpl.rcParams['figure.facecolor'] = (1,1,1,1)
#   mpl.rcParams["savefig.facecolor"] = (1,1,1,1)
#   mpl.rcParams['font.family'] = 'serif'
#   mpl.rcParams['font.size'] = 15.0
#   mpl.rcParams['mathtext.rm'] = 'serif'
#   mpl.rcParams['figure.dpi'] = 150
#   mpl.rcParams['axes.labelsize'] = 'large' # fontsize of the x any y labels
#   Pmin = 150    # XXX upper limit
#   Pmax = 1000   # XXX lower limit
#   LG.debug('Checking units')
#   if p.attrs['units'] != 'hPa':
#      LG.critical('P wrong units')
#      exit()
#   if tc.attrs['units'] != 'degC':
#      LG.critical('Tc wrong units')
#      exit()
#   if tdc.attrs['units'] != 'degC':
#      LG.critical('Tdc wrong units')
#      exit()
#   if t0.attrs['units'] != 'degC':
#      LG.critical('T0 wrong units')
#      exit()
#   if type(u) != type(None) and type(v) != type(None):
#      if u.attrs['units'] != 'm s-1':
#         LG.critical('Wind wrong units')
#         exit()
#   LG.info('Inside skewt plot')
#   p = p.values
#   tc = tc.values
#   tdc = tdc.values
#   t0 = t0.mean().values
#   u = u.values * 3.6  # km/h
#   v = v.values * 3.6  # km/h
#   ############
#   LG.warning('Interpolating sounding variables')
#   Ninterp = 250
#   ps = np.linspace(np.max(p),np.min(p), Ninterp)
#   ftc = interp1d(p,tc)
#   ftdc = interp1d(p,tdc)
#   fu = interp1d(p,u)
#   fv = interp1d(p,v)
#   p = ps
#   tc = ftc(ps)
#   tdc = ftdc(ps)
#   u = fu(ps)
#   v = fv(ps)
#   ############
#   # Grid plot
#   LG.info('creating figure')
#   fig = plt.figure(figsize=(11, 12))
#   LG.info('created figure')
#   LG.info('creating axis')
#   gs = gridspec.GridSpec(1, 3, width_ratios=[6,0.4,1.8])
#   fig.subplots_adjust(wspace=0.,hspace=0.)
#   # ax1 = plt.subplot(gs[1:-1,0])
#   # Adding the "rotation" kwarg will over-ride the default MetPy rotation of
#   # 30 degrees for the 45 degree default found in NCL Skew-T plots
#   LG.info('created axis')
#   LG.info('Creatin SkewT')
#   skew = SkewT(fig, rotation=45, subplot=gs[0,0])
#   ax = skew.ax
#   LG.info('Created SkewT')

#   if len(latlon) > 0:
#       ax.text(0,1, latlon, va='top', ha='left', color='k',
#                     fontsize=12, bbox=dict(boxstyle="round",
#                                            ec=None, fc=(1., 1., 1., 0.9)),
#                     zorder=100, transform=ax.transAxes)
#   # Plot the data, T and Tdew vs pressure
#   skew.plot(p, tc,  'C3')
#   skew.plot(p, tdc, 'C0')
#   LG.info('plotted dew point and sounding')

#   # LCL
#   lcl_p, lcl_t = mpcalc.lcl(p[0]*units.hPa,
#                                              t0*units.degC,
#                                              tdc[0]*units.degC)
#   skew.plot(lcl_p, lcl_t, 'k.')
#   lcl_p = lcl_p.magnitude
#   lcl_t = lcl_t.magnitude

#   # Calculate the parcel profile  #XXX units workaround
#   parcel_prof = mpcalc.parcel_profile(p* units.hPa,
#                                       t0 * units.degC,
#                                       tdc[0]* units.degC).to('degC')
#   # Plot cloud base
#   # p_base, t_base = find_cross(parcel_prof.magnitude, tc, p, tc, interp=True)
#   # p_base = np.max([lcl_pressure.magnitude, p_base])
#   # t_base = np.max([lcl_temperature.magnitude, t_base])
#   p_base, t_base = get_cloud_base(parcel_profile, p, tc, lcl_p, lcl_t)
#   parcel_cross = p_base, t_base
#   m_base = mpcalc.pressure_to_height_std(np.array(p_base)*units.hPa)
#   m_base = m_base.to('m').magnitude
#   skew.plot(p_base, t_base, 'C3o', zorder=100)
#   skew.ax.text(t_base, p_base, f'{m_base:.0f}m',ha='left')

#   # Plot the parcel profile as a black line
#   skew.plot(p, parcel_prof, 'k', linewidth=1)
#   LG.info('plotted parcel profile')

#   # shade CAPE and CIN
#   skew.shade_cape(p* units.hPa,
#                   tc * units.degC, parcel_prof)
#   skew.shade_cin(p * units.hPa,
#                  tc * units.degC,
#                  parcel_prof,
#                  tdc * units.degC)
#   LG.info('plotted CAPE and CIN')

#   ## Clouds ############
#   ps = np.linspace(np.max(p),np.min(p),500)
#   tcs = interp1d(p,tc)(ps)
#   tds = interp1d(p,tdc)(ps)
#   x0 = 0.3
#   t = 0.2
#   overcast = fermi(tcs-tds, x0=x0,t=t)
#   overcast = overcast/fermi(0, x0=x0,t=t)
#   cumulus = np.where((p_base>ps) & (ps>parcel_cross[0]),1,0)
#   cloud = np.vstack((overcast,cumulus)).transpose()
#   ax_cloud = plt.subplot(gs[0,1],sharey=ax, zorder=-1)
#   ax_cloud.imshow(cloud,origin='lower',extent=[0,2,p[0],p[-1]],aspect='auto',cmap='Greys',vmin=0,vmax=1)
#   ax_cloud.text(0,0,'O',transform=ax_cloud.transAxes)
#   ax_cloud.text(0.5,0,'C',transform=ax_cloud.transAxes)
#   ax_cloud.set_xticks([])
#   plt.setp(ax_cloud.get_yticklabels(), visible=False)
#   #####################

#   if type(u) != type(None) and type(v) != type(None):
#      LG.info('Plotting wind')
#      ax_wind = plt.subplot(gs[0,2], sharey=ax, zorder=-1)
#      ax_wind.yaxis.tick_right()
#      # ax_wind.xaxis.tick_top()
#      wspd = np.sqrt(u*u + v*v)
#      ax_wind.scatter(wspd, p, c=p, cmap=HEIGHTS, zorder=10)
#      gnd = mpcalc.pressure_to_height_std(np.array(p[0])*units.hPa)
#      gnd = gnd.to('m')
#      ### Background colors ##
#      #for i,c in enumerate(WindSpeed.colors):
#      #   rect = Rectangle((i*4, 150), 4, 900,  color=c, alpha=0.5,zorder=-1)
#      #   ax_wind.add_patch(rect)
#      #########################
#      # X axis
#      ax_wind.set_xlim(0,56)
#      ax_wind.set_xlabel('Wspeed (km/h)')
#      ax_wind.set_xticks([0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56])
#      ax_wind.set_xticklabels(['0','','8','','16','','24','32','40','48','56'], fontsize=11, va='center')
#      # Y axis
#      plt.setp(ax_wind.get_yticklabels(), visible=False)
#      ax_wind.grid()
#      def p2h(x):
#         """
#         x in hPa
#         """
#         y = mpcalc.pressure_to_height_std(np.array(x)*units.hPa)
#         # y = y.metpy.convert_units('m')
#         y = y.to('m')
#         return y.magnitude
#      def h2p(x):
#         """
#         x in m
#         """
#         # x = x.values
#         y = mpcalc.height_to_pressure_std(np.array(x)*units.m)
#         # y = y.metpy.convert_units('hPa')
#         y = y.to('hPa')
#         return y.magnitude

#      # Duplicate axis in meters
#      ax_wind_m = ax_wind.secondary_yaxis(1.02,functions=(p2h,h2p))
#      ax_wind_m.set_ylabel('height (m)')
#      # XXX Not working
#      ax_wind_m.yaxis.set_major_formatter(ScalarFormatter())
#      ax_wind_m.yaxis.set_minor_formatter(ScalarFormatter())
#      #################
#      ax_wind_m.set_color('red')
#      ax_wind_m.tick_params(colors='red',size=7, width=1, which='both')  # 'both' refers to minor and major axes
#      # Hodograph
#      ax_hod = inset_axes(ax_wind, '110%', '30%', loc=1)
#      ax_hod.set_yticklabels([])
#      ax_hod.set_xticklabels([])
#      L = 80
#      ax_hod.text(  0, L-5,'N', horizontalalignment='center',
#                               verticalalignment='center')
#      ax_hod.text(L-5,  0,'E', horizontalalignment='center',
#                               verticalalignment='center')
#      ax_hod.text(-(L-5),0 ,'W', horizontalalignment='center',
#                               verticalalignment='center')
#      ax_hod.text(  0,-(L-5),'S', horizontalalignment='center',
#                               verticalalignment='center')
#      h = Hodograph(ax_hod, component_range=L)
#      h.add_grid(increment=20)
#      h.plot_colormapped(-u, -v, p, cmap=HEIGHTS)  #'viridis_r')  # Plot a line colored by pressure (altitude)
#      LG.info('Plotted wind')

#      ## Plot only every n windbarb
#      n = Ninterp//30
#      inds, = np.where(p>Pmin)
#      # break_p = 25
#      # inds_low = inds[:break_p]
#      # inds_high = inds[break_p:]
#      # inds = np.append(inds[::n], inds_high)
#      inds = inds[::n]
#      skew.plot_barbs(pressure=p[inds], # * units.hPa,
#                      u=u[inds],
#                      v=v[inds],
#                      xloc=0.985, # fill_empty=True,
#                      sizes=dict(emptybarb=0.075, width=0.1, height=0.2))

#   # Add relevant special lines
#   ## cumulus base
#   skew.ax.axhline(p_base, color=(0.5,0.5,0.5), ls='--')
#   ax_wind.axhline(p_base,c=(0.5,0.5,0.5),ls='--')
#   ax_cloud.axhline(p_base,c=(0.5,0.5,0.5),ls='--')
#   ## cumulus top
#   skew.ax.axhline(parcel_cross[0], color=(.75,.75,.75), ls='--')
#   ax_wind.axhline(parcel_cross[0], color=(.75,.75,.75), ls='--')
#   ax_cloud.axhline(parcel_cross[0], color=(.75,.75,.75), ls='--')

#   # Ground
#   skew.ax.axhline(p[0],c='k',ls='--')
#   ax_wind.axhline(p[0],c='k',ls='--')
#   ax_cloud.axhline(p[0],c='k',ls='--')
#   ax_wind.text(56,p[0],f'{int(gnd.magnitude)}m',horizontalalignment='right')
#   # Choose starting temperatures in Kelvin for the dry adiabats
#   LG.info('Plot adiabats, and other grid lines')
#   skew.ax.text(t0,Pmax,f'{t0:.1f}°C',va='bottom',ha='left')
#   skew.ax.axvline(t0, color=(0.5,0.5,0.5),ls='--')
#   t0 = units.K * np.arange(243.15, 473.15, 10)
#   skew.plot_dry_adiabats(t0=t0, linestyles='solid', colors='gray', linewidth=1)

#   # Choose temperatures for moist adiabats
#   t0 = units.K * np.arange(281.15, 306.15, 4)
#   msa = skew.plot_moist_adiabats(t0=t0,
#                                  linestyles='solid',
#                                  colors='lime',
#                                  linewidths=.75)

#   # Choose mixing ratios
#   w = np.array([0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020]).reshape(-1, 1)

#   # Choose the range of pressures that the mixing ratio lines are drawn over
#   p_levs = units.hPa * np.linspace(1000, 400, 7)
#   skew.plot_mixing_lines(mixing_ratio=w, pressure=p_levs,
#                          linestyle='dotted',linewidths=1, colors='lime')

#   LG.info('Plotted adiabats, and other grid lines')

#   skew.ax.set_ylim(Pmax, Pmin)
#   skew.ax.set_xlim(-20, 43)

#   # Change the style of the gridlines
#   ax.grid(True, which='major', axis='both',
#            color='tan', linewidth=1.5, alpha=0.5)

#   ax.set_xlabel("Temperature (C)")
#   ax.set_ylabel("P (hPa)")
#   if len(title) == 0:
#      title = f"{(date).strftime('%d/%m/%Y-%H:%M')} (local time)"
#   ax.set_title(title, fontsize=20)
#   if show: plt.show()
#   LG.info('saving')
#   fig.savefig(fout, bbox_inches='tight', pad_inches=0.1, dpi=150, quality=90)
#   LG.info('saved')
#   plt.close('all')
