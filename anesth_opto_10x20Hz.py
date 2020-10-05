# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:36:40 2020

@author: akees
"""

# GluK2 anesthetized optogenetic stimulation experiments
# opto stim in DG, record in CA3
# protocol: opto_10x20Hz
# loads from .nix 1250Hz files


#%% import modules

import neo  
from neo import NixIO
import numpy as np
import matplotlib.pyplot as plt
import os
import quantities as pq
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm
from itertools import compress
from scipy import signal
from scipy import stats
#import intanutil
#import load_intan_rhd_format
#from load_intan_rhd_format import read_data
#runfile(r'C:\Users\akees\Documents\Python Scripts\Intan\load_intan_rhd_format\load_intan_rhd_format.py')


     
    
#%% load data

# load the excel sheet list of data
sheet_name = 'opto_10x20Hz'
data_folder = r'C:\Users\akees\Documents\Ashley\Datasets\GluK2\anesthetized\opto_10x20Hz\1250Hz'

data_list = pd.read_excel(r"C:\Users\akees\Documents\Ashley\Analysis\GluK2\Data_to_Analyze_GluK2.xlsx",
                          sheet_name=sheet_name)

# create the list of data dictionaries
data = [{} for i in np.arange(data_list.shape[0])]

#for i in np.arange(len(data)):
for i in tqdm(np.arange(len(data))):
    
    # load the nix file
    file = os.path.join(data_folder, data_list['Filename'][i] + '.nix')
    io = NixIO(file, 'ro')
    bl = io.read_block()
    
    data[i]['bl'] = bl


#%% set number of channels (assume same for all sessions)
    
nb_channel = data[0]['bl'].segments[0].analogsignals[0].magnitude.shape[1]

# %% make stim and train blocks


# make slices around the trains (or pulses, depending on the protocol)
# define how much before and after epoch to take
# version: make sure all are same number of samples
e_win = np.array([-10, 30])*pq.ms
e_ind = 0  # determines whether you slice over stims or trains
for i in np.arange(len(data)):
    num_seg = len(data[i]['bl'].segments[0].epochs[e_ind])
    s_start = data[i]['bl'].segments[0].epochs[e_ind].times + e_win[0]
    start_ind = np.searchsorted(data[i]['bl'].segments[0].analogsignals[0].times, s_start)
    s_start = data[i]['bl'].segments[0].analogsignals[0].times[start_ind]
    dur_ind = int(np.diff(e_win).rescale(pq.s)*data[i]['bl'].segments[0].analogsignals[0].sampling_rate)
    s_stop = data[i]['bl'].segments[0].analogsignals[0].times[start_ind+dur_ind]
    bl_stim = neo.Block(name='stim slices')
    for s in np.arange(num_seg):
        seg = data[i]['bl'].segments[0].time_slice(s_start[s], s_stop[s])
        bl_stim.segments.append(seg)
    data[i]['bl_stim'] = bl_stim
    # make triggered timestamps
    t_ts = bl_stim.segments[0].analogsignals[0].times - data[i]['bl'].segments[0].epochs[e_ind].times[0]
    data[i]['stim_ts'] = t_ts.rescale(pq.ms)

# version: make sure all are same number of samples
e_win = np.array([-10, 540])*pq.ms
e_ind = 1  # determines whether you slice over stims or trains
for i in np.arange(len(data)):
    num_seg = len(data[i]['bl'].segments[0].epochs[e_ind])
    s_start = data[i]['bl'].segments[0].epochs[e_ind].times + e_win[0]
    start_ind = np.searchsorted(data[i]['bl'].segments[0].analogsignals[0].times, s_start)
    s_start = data[i]['bl'].segments[0].analogsignals[0].times[start_ind]
    dur_ind = int(np.diff(e_win).rescale(pq.s)*data[i]['bl'].segments[0].analogsignals[0].sampling_rate)
    s_stop = data[i]['bl'].segments[0].analogsignals[0].times[start_ind+dur_ind]
    bl_stim = neo.Block(name='train slices')
    for s in np.arange(num_seg):
        seg = data[i]['bl'].segments[0].time_slice(s_start[s], s_stop[s])
        bl_stim.segments.append(seg)
    data[i]['bl_train'] = bl_stim
    # make triggered timestamps
    t_ts = bl_stim.segments[0].analogsignals[0].times - data[i]['bl'].segments[0].epochs[e_ind].times[0]
    data[i]['train_ts'] = t_ts.rescale(pq.ms)


#%% figure parameters

viridis_16 = ['#481567', '#482677', '#453781', '#404788',
              '#39568C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
              '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C667',
              '#73D055', '#95D840']
viridis_32 = np.flip(np.reshape(np.transpose(np.array([viridis_16, viridis_16])), [32,]))


offset = 0.1
#offset = 0.5  # for 1 Hz
#offset = 0.4  # for 20 Hz
offset_matrix = np.flip(np.arange(offset, offset*(nb_channel+1), offset)*pq.mV)
viridis_ordered = viridis_32
#offset_matrix = offset_matrix[bl.annotations['depth_order']]
#viridis_ordered = viridis_32[bl.annotations['depth_order']]

#magma_7 = ['#000000', '#6B116F', '#981D69', '#C92D59', '#ED504A', '#FA8657', '#FBC17D']
magma_4 = ['#000000', '#981D69', '#ED504A', '#FBC17D']

c_lgry = [0.75, 0.75, 0.75]
c_mgry = [0.5, 0.5, 0.5]
c_dgry = [0.25, 0.25, 0.25]
c_wht = [1, 1, 1]
c_blk = [0, 0, 0]

c_opto = [0.01, 0.66, 0.95]

# set style defaults
mpl.rcParams['font.size'] = 16
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['boxplot.whiskerprops.linestyle'] = '-'
mpl.rcParams['patch.force_edgecolor'] = True
mpl.rcParams['patch.facecolor'] = 'b'


# set figure output folder
fig_folder = os.path.join(r'C:\Users\akees\Documents\Ashley\Analysis\GluK2\anesthetized\opto_stim',
                          sheet_name)


#%% figure - visualize the stim pattern

i = 0
stim_times = data[i]['bl'].segments[0].epochs[0].times
fig, ax = plt.subplots(1, 1, figsize=[4, 1.5])
for s in stim_times:
    ax.axvline(s, color=c_opto, alpha=0.5, linewidth=2)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim([stim_times[0]-0.1*pq.s, stim_times[9]+0.4*pq.s])
plt.tight_layout()

# %% figure - raw traces

start = 1000
ds_idx = np.arange(start, start + 20000)
plt.figure()
for c in np.arange(nb_channel):
    plt.plot(bl.segments[0].analogsignals[0].times[ds_idx],
         (bl.segments[0].analogsignals[0]+offset_matrix[c]).magnitude[ds_idx, c],
         color=viridis_ordered[c])
plt.tight_layout() 


#%% figure: average trace over all stim (every frequency)
# works for opto_LFF, opto_10x20Hz

# plot just the average over all trains and save
for i in np.arange(len(data)):
    t_ts = data[i]['train_ts']
    bl_stim = data[i]['bl_train']
    fig, ax = plt.subplots(1, 1, figsize=[6, 7])   
    # make the average
    all_traces = np.array([s.analogsignals[0].magnitude for s in bl_stim.segments])
    try:
        grand_mean = np.mean(all_traces, axis=0)*pq.uV    
    except ValueError:
        lengths = np.array([s.analogsignals[0].magnitude.shape[0] for s in bl_stim.segments])
        v, c = np.unique(lengths, return_counts=True)
        # don't include the ones that don't match
        idx = np.where(lengths == v[np.argmax(c)])[0]   
        grand_mean = np.mean(all_traces[idx], axis=0)*pq.uV 
    for c in np.arange(nb_channel):
        plt.plot(t_ts, grand_mean[:, c]+offset_matrix[c], color=viridis_ordered[c], zorder=2)
    t0 = data[i]['bl_stim'].segments[0].analogsignals[0].times[0].rescale(pq.ms)
    t0 = t0-e_win[0]
    for s in np.arange(data[i]['bl_train'].segments[0].epochs[0].times.size):
        t = data[i]['bl_train'].segments[0].epochs[0].times[s].rescale(pq.ms)
        d = data[i]['bl_train'].segments[0].epochs[0].durations[s].rescale(pq.ms)
        ax.axvspan(t-t0, t+d-t0, ymin=0, ymax=1, color='c', alpha=0.25, zorder=1)
    ax.spines['left'].set_bounds(0, 100)
    ax.set_yticks([])
    ax.set_xlim([-10, 500])
    ax.set_xlabel('ms')
    ax.set_title(data_list['Filename'][i] + '\n(scalebar = 100 uV)')
    fig.tight_layout()
    #plt.savefig(os.path.join(fig_folder, data_list['Filename'][i] + '.png'), transparent=True)
    plt.savefig(os.path.join(fig_folder, 'ex.png'), transparent=True)
    plt.close()



# %% quantify fiber volley and fEPSP on average traces (opto_10x20Hz only)

num_stim = len(data[0]['bl_train'].segments[0].epochs[0])
stim_num = np.arange(num_stim)
num_train = len(data[0]['bl_train'].segments)
train_num = np.arange(num_train)

for i in np.arange(len(data)):
    all_traces = np.full((num_train, num_stim, data[i]['stim_ts'].size, nb_channel), np.nan)
    bl_stim = data[i]['bl_stim']
    for s in np.arange(len(data[i]['bl_stim'].segments)):
        label = bl_stim.segments[s].epochs[0].labels[0]
        label = label.astype(str)
        _idx = label.find('_')
        train_id = int(label[:_idx])
        stim_id = int(label[_idx+1:])
        all_traces[train_id, stim_id, :, :] = bl_stim.segments[s].analogsignals[0].magnitude
    data[i]['traces'] = all_traces


# find the channel with the biggest downward fEPSP in each stim of the average train
sl_channel_stim = np.full((len(data), num_stim), np.nan)
for i in np.arange(len(data)):
    ts = data[i]['stim_ts']
    mean_train = np.nanmean(data[i]['traces'], axis=0)
    idx0 = np.searchsorted(ts, 0)
    idx1 = np.searchsorted(ts, 6)
    idx2 = np.searchsorted(ts, 20)
    fEPSP_amp = np.min(mean_train[:, idx1:idx2, :], axis=1) - mean_train[:, idx0, :]
    sl_channel_stim[i, :] = np.argmin(fEPSP_amp, axis=1)
sl_channel_stim = sl_channel_stim.astype(int)

# find the channel with the biggest cumulative fEPSP over the 10 stim
sl_channel_sum = np.full(len(data), np.nan)
for i in np.arange(len(data)):
    ts = data[i]['stim_ts']
    mean_train = np.nanmean(data[i]['traces'], axis=0)
    idx0 = np.searchsorted(ts, 0)
    idx1 = np.searchsorted(ts, 6)
    idx2 = np.searchsorted(ts, 20)
    fEPSP_amp = np.min(mean_train[:, idx1:idx2, :], axis=1) - mean_train[:, idx0, :]
    sl_channel_sum[i] = np.argmin(np.sum(fEPSP_amp, axis=0))
sl_channel_sum = sl_channel_sum.astype(int)

# find the channel with the biggest facilitation over the 10 stim
sl_channel_fac = np.full(len(data), np.nan)
for i in np.arange(len(data)):
    ts = data[i]['stim_ts']
    mean_train = np.nanmean(data[i]['traces'], axis=0)
    idx0 = np.searchsorted(ts, 0)
    idx1 = np.searchsorted(ts, 6)
    idx2 = np.searchsorted(ts, 20)
    fEPSP_amp = np.min(mean_train[:, idx1:idx2, :], axis=1) - mean_train[:, idx0, :]
    sl_channel_fac[i] = np.argmax(fEPSP_amp[9, :]/fEPSP_amp[0, :])
sl_channel_fac = sl_channel_fac.astype(int)

## looks like the 3rd stim gives a decent estimate of the best channel
#sl_channel = sl_channel_stim[:, 2]
# or just set by hand
sl_channel = np.array(data_list['selected channel'])

# plot the 0/3/6/9 stim one on top of the other
fig, ax = plt.subplots(3, 7, figsize=[18, 9], sharex=True, sharey=True,
                       constrained_layout=True)
axs = ax.flat  
for i in np.arange(len(data)):
    t_ts = data[i]['stim_ts']
    mean_train = np.nanmean(data[i]['traces'], axis=0)
    c = sl_channel[i]
    idx1 = np.searchsorted(ts, 6)
    idx2 = np.searchsorted(ts, 20)
    for s in np.arange(4):
        axs[i].plot(t_ts, mean_train[s*3, :, c], color=magma_4[s])
        idx = np.argmin(mean_train[s*3, idx1:idx2, c])
        axs[i].scatter(ts[idx+idx1], mean_train[s*3, idx+idx1, c], color=magma_4[s])
    t0 = data[i]['bl_stim'].segments[0].analogsignals[0].times[0].rescale(pq.ms)
    t0 = t0-e_win[0]
    t = data[i]['bl_stim'].segments[0].epochs[0].times[0].rescale(pq.ms)
    d = data[i]['bl_stim'].segments[0].epochs[0].durations[0].rescale(pq.ms)
    axs[i].axvspan(t-t0, t+d-t0, ymin=0, ymax=1, color='c', alpha=0.1, zorder=1)
    axs[i].set_title(data_list['Filename'][i] + ', ch' + str(sl_channel[i]), fontsize=10)
axs[0].set_xlim([-10, 30])
axs[14].set_xlabel('ms')
axs[14].set_ylabel('uV')


# %% plot the average trains

num_train = len(data[0]['bl_train'].segments)
train_num = np.arange(num_train)

for i in np.arange(len(data)):
    all_trains = np.full((num_train, data[i]['train_ts'].size, nb_channel), np.nan)
    bl_stim = data[i]['bl_train']
    for s in np.arange(len(data[i]['bl_train'].segments)):
        all_trains[s, :, :] = bl_stim.segments[s].analogsignals[0].magnitude
    data[i]['trains'] = all_trains

# plot the average train for each session
fig, ax = plt.subplots(7, 3, figsize=[18, 9], sharex=True, constrained_layout=True)
axs = ax.flat  
for i in np.arange(len(data)):
    t_ts = data[i]['train_ts']
    mean_train = np.nanmean(data[i]['trains'], axis=0)
    c = sl_channel[i]
    if data_list['Cre'][i] == 'F':
        color=c_blk
    if data_list['Cre'][i] == 'T':
        color=viridis_16[8]
    axs[i].plot(t_ts, mean_train[:, c], color=color)
    t0 = data[i]['bl_train'].segments[0].analogsignals[0].times[0].rescale(pq.ms)
    t0 = t0-e_win[0]
    for s in np.arange(data[i]['bl_train'].segments[0].epochs[0].times.size):
        t = data[i]['bl_train'].segments[0].epochs[0].times[s].rescale(pq.ms)
        d = data[i]['bl_train'].segments[0].epochs[0].durations[s].rescale(pq.ms)
        axs[i].axvspan(t-t0, t+d-t0, ymin=0, ymax=1, color='c', alpha=0.1, zorder=1)
    axs[i].set_title(data_list['Filename'][i] + ', ch' + str(sl_channel[i]), fontsize=10)
axs[18].set_xlabel('ms')
axs[18].set_ylabel('uV')

# plot the average train for each session
# version: one example session
i = 7
fig, axs = plt.subplots(1, 1, figsize=[6, 4]) 
t_ts = data[i]['train_ts']
mean_train = np.nanmean(data[i]['trains'], axis=0)
c = sl_channel[i]
axs.plot(t_ts, mean_train[:, c], color=c_blk)
t0 = data[i]['bl_train'].segments[0].analogsignals[0].times[0].rescale(pq.ms)
t0 = t0-e_win[0]
for s in np.arange(data[i]['bl_train'].segments[0].epochs[0].times.size):
    t = data[i]['bl_train'].segments[0].epochs[0].times[s].rescale(pq.ms)
    d = data[i]['bl_train'].segments[0].epochs[0].durations[s].rescale(pq.ms)
    axs.axvspan(t-t0, t+d-t0, ymin=0, ymax=1, color='c', alpha=0.1, zorder=1)
axs.set_title(data_list['Filename'][i] + ', ch' + str(sl_channel[i]), fontsize=10)
axs.spines['left'].set_bounds(-100, -300)
axs.set_yticks([])
axs.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
axs.set_xticklabels([0, '', 100, '', 200, '', 300, '', 400, '', 500])
axs.set_xlabel('ms')
fig.tight_layout()



# %% find the fEPSP amplitude for each stim (on only the selected channel, average train)


fEPSP_amp = np.full((len(data), num_stim), np.nan)
for i in np.arange(len(data)):
    c = sl_channel[i]
    mean_train = np.nanmean(data[i]['traces'], axis=0)[:, :, c]
    ts = data[i]['stim_ts']
    ts = ts -ts[0] + e_win[0]
    idx0 = np.searchsorted(ts, 0)
    idx1 = np.searchsorted(ts, 6)
    idx2 = np.searchsorted(ts, 20)
    fEPSP_amp[i, :] = np.min(mean_train[:, idx1:idx2], axis=1) - mean_train[:, idx0]

keep_sessions = fEPSP_amp[:, 0] < -30

# plot % change over stim (baseline is first stim in the train)
fEPSP_amp_perc = 100*(fEPSP_amp/np.atleast_2d(fEPSP_amp[:, 0]).T)
perc_cre = fEPSP_amp_perc[np.logical_and(keep_sessions, data_list['Cre'] == 'T'), :]
perc_ctrl = fEPSP_amp_perc[np.logical_and(keep_sessions, data_list['Cre'] == 'F'), :]
fig, ax = plt.subplots(1, 1)
for i in np.arange(len(data)):
    if data_list['Cre'][i] == 'T':
        color=viridis_16[8]
    else:
        color=c_blk
    if keep_sessions[i]:
        ax.plot(np.arange(fEPSP_amp.shape[1])+1, fEPSP_amp_perc[i, :],
                color=color, alpha=0.15, zorder=2)
ax.plot(np.arange(fEPSP_amp.shape[1])+1, np.nanmean(perc_cre, axis=0),
                color=viridis_16[8], zorder=3)
ax.plot(np.arange(fEPSP_amp.shape[1])+1, np.nanmean(perc_ctrl, axis=0),
                color=c_dgry, zorder=3)
ax.axhline(100, color=c_blk, zorder=1, linestyle='--')
ax.set_title('all sessions, (blue=Cre)')
ax.set_ylabel('fEPSP amplitude (% change from first stim)')
ax.set_xlabel('stim number')
plt.tight_layout()

# plot first stim amplitude, compare Cre and nonCre
fig, ax = plt.subplots(1, 1)
for i in np.arange(len(data)):
    if data_list['Cre'][i] == 'T':
        color=viridis_16[8]
        x = 2
    else:
        color=c_blk
        x = 1
    if keep_sessions[i]:
        ax.scatter(x, fEPSP_amp[i, 0], color=color)
ax.set_title('first stimulus amplitude')
ax.set_ylabel('fEPSP amplitude (uV)')
ax.set_xticks([1, 2])
ax.set_xticklabels(['ctl', 'cre'])
ax.set_xlim([0, 3])
plt.tight_layout()









