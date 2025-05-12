#!/usr/bin/python3
# -*- coding: UTF-8 -*-

##############################################################################
from calc_data import CalcData
import utils as ut

folder = '../../Documents/storage/WRFOUT/Spain6_1'
date = '2025-05-10'
fname = f'{folder}/wrfout_d02_{date}_07:00:00'
fname = f'{folder}/wrfout_d02_{date}_08:00:00'
fname = f'{folder}/wrfout_d02_{date}_09:00:00'
fname = f'{folder}/wrfout_d02_{date}_10:00:00'
fname = f'{folder}/wrfout_d02_{date}_11:00:00'
fname = f'{folder}/wrfout_d02_{date}_12:00:00'
# fname = f'{folder}/wrfout_d02_{date}_13:00:00'
# fname = f'{folder}/wrfout_d02_{date}_14:00:00'
# fname = f'{folder}/wrfout_d02_{date}_15:00:00'
# fname = f'{folder}/wrfout_d02_{date}_16:00:00'
# fname = f'{folder}/wrfout_d02_{date}_17:00:00'
# fname = f'{folder}/wrfout_d02_{date}_18:00:00'
# fname = f'{folder}/wrfout_d02_{date}_19:00:00'

domain = ut.get_domain(fname)
date = ut.file2date(fname)


from time import time


output_folder,plots_folder,data_folder = ut.get_folders()
told = time()
A = CalcData(fname, OUT_folder=plots_folder, DATA_folder=data_folder)
print(f'Process WRF: {time()-told:.5f}s')

# import numpy as np
# import matplotlib.pyplot as plt
# try: plt.style.use('mystyle')
# except: pass

# def rh_log_threshold(nz, rh_top=0.9, rh_bottom=1):
#    """Create a log-scaled RH threshold profile."""
#    levels = np.arange(nz) #[::-1]  # from top (high index) to bottom (0)
#    log_scaled = np.log1p(levels) / np.log1p(levels[-1])  # Normalize log scale to [0, 1]
#    return rh_top + (rh_bottom - rh_top) * (1 - log_scaled)

# def rh_lin_threshold(nz, rh_top=0.85, rh_bottom=0.99):
#    threshold = np.linspace(rh_bottom, rh_top, rh.shape[0]) # [:, np.newaxis, np.newaxis]
#    return threshold


# def rh_to_cloud_prob(rh, threshold=0.95, slope=30):
#    """
#    Maps RH to a 0–1 probability with a sharp transition near the threshold
#    """
#    return 1 / (1 + np.exp(-slope * (rh - threshold)))

# i,j = 108, 192
# Z = A.wrf_vars['heights'][:,i,j]
# rh = A.wrf_vars['rh'][:,i,j] / 100
# nz = rh.shape[0]
# th_log = rh_log_threshold(nz)
# # threshold = rh_log_threshold(nz)[:, np.newaxis, np.newaxis]  # shape (nz, 1, 1)
# # cloud_mask = rh > threshold
# overcast_log = rh_to_cloud_prob(rh, th_log, slope=60)


# from matplotlib import gridspec
# fig = plt.figure()
# gs = gridspec.GridSpec(2, 1)
# fig.subplots_adjust() #wspace=0.1,hspace=0.1)
# ax0 = plt.subplot(gs[0,0])
# ax1 = plt.subplot(gs[1,0])

# ax0.plot(Z,rh)
# ax0.plot(Z,overcast_log)

# rh = np.linspace(0,1,100)
# nz = rh.shape[0]
# th_log = rh_log_threshold(nz)
# th_lin = rh_lin_threshold(nz)
# overcast_50 = rh_to_cloud_prob(rh, th_log, slope=50)
# overcast_60 = rh_to_cloud_prob(rh, th_log, slope=60)
# ax1.plot(rh,overcast_50)
# ax1.plot(rh,overcast_60)
# ax1.set_ylabel('overcast %')
# ax1.set_xlabel('rh')


# fig.tight_layout()
# plt.show()

# exit()




lat, lon = 41.1, -3.6
# # index
# i,j = 108, 192

from plots.sounding import skew_t_plot
told = time()
skew_t_plot(A, lat,lon,fout=f'sounding_{A.tail}.png')
print(f'Sounding plot: {time()-told:.5f}s')




from meteogram_writer import make_meteogram_timestep, append_to_meteogram
ds = make_meteogram_timestep(A, lat, lon)
append_to_meteogram(ds, f"meteogram_{lat}_{lon}.nc")

## meteogram_watchr.py
from plots.meteogram import plot_meteogram
plot_meteogram(f"meteogram_{lat}_{lon}.nc",fout=f'meteogram_{A.tail_d}.png')





exit()
# # output_folder = expanduser( P['system']['output_folder'] )
# # plots_folder = expanduser( P['system']['plots_folder'] )
# # data_folder = expanduser( P['system']['data_folder'] )
# ut.check_directory(output_folder,True)
# ut.check_directory(output_folder+'/processed',False)
# ut.check_directory(plots_folder,False)
# ut.check_directory(data_folder,False)
            
zooms = ut.get_zooms('zooms.ini',domain=domain)

import numpy as np
print(np.min(hcrit:=A.drjack_vars['hcrit'].values))
print(np.max(hcrit))

import matplotlib.pyplot as plt
try: plt.style.use('mystyle')
except: pass
fig, ax = plt.subplots()
ax.contourf(hcrit, vmin=300, vmax=2500)
fig.tight_layout()
plt.show()

