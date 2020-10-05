# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:53:31 2020

@author: akees
"""


# GluK2 anesthetized optogenetic stimulation experiments
# opto stim in DG, record in CA3
# protocol: opto_LFF
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
sheet_name = 'opto_LFF'
data_folder = r'C:\Users\akees\Documents\Ashley\Datasets\GluK2\anesthetized\opto_LFF\1250Hz'

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

# %% make stim block


# make slices around the trains (or pulses, depending on the protocol)
# define how much before and after epoch to take
e_win = np.array([-10, 150])*pq.ms
e_ind = 0  # determines whether you slice over stims or trains
for i in tqdm(np.arange(len(data))):
    e_list = [d.name for d in data[i]['bl'].segments[0].epochs]
    num_seg = len(data[i]['bl'].segments[0].epochs[e_ind])
    s_start = data[i]['bl'].segments[0].epochs[e_ind].times + e_win[0]
    s_stop = s_start + data[i]['bl'].segments[0].epochs[e_ind].durations - e_win[0] + e_win[1]
    bl_stim = neo.Block(name='stim slices')
    for s in np.arange(num_seg):
        seg = data[i]['bl'].segments[0].time_slice(s_start[s], s_stop[s])
        bl_stim.segments.append(seg)
    data[i]['bl_stim'] = bl_stim
    # make triggered timestamps
    t_ts = bl_stim.segments[0].analogsignals[0].times - data[i]['bl'].segments[0].epochs[e_ind].times[0]
    data[i]['t_ts'] = t_ts.rescale(pq.ms)

# make slices around the trains (or pulses, depending on the protocol)
# define how much before and after epoch to take
#e_win = np.array([-10, 140])*pq.ms  # have this match the one above, so t_ts can be the same
e_ind = 0  # determines whether you slice over stims or trains
# find the different frequencies
stim_freq = np.unique(data[0]['bl'].segments[0].epochs[0].labels)
# create a block for each stim_freq, where the segments are each stim of that frequency
for i in tqdm(np.arange(len(data))):
    for f in np.arange(stim_freq.size):
        stim_epochs = data[i]['bl'].filter(objects='Epoch', name='opto_LFF')[0]
        mask = stim_epochs.labels == stim_freq[f]
        freq_epochs = stim_epochs[mask]
        num_seg = freq_epochs.size
        s_start = freq_epochs.times + e_win[0]
        s_stop = s_start + freq_epochs.durations - e_win[0] + e_win[1]
        bl_freq = neo.Block(name=stim_freq[f])
        for s in np.arange(num_seg):
            seg = data[i]['bl'].segments[0].time_slice(s_start[s], s_stop[s])
            bl_freq.segments.append(seg)
        data[i]['bl_'+stim_freq[f].astype(str)] = bl_freq

disp_freq = ['0.1 Hz', '0.5 Hz', '1.0 Hz', '3.0 Hz']



#%% figure parameters

viridis_16 = ['#481567', '#482677', '#453781', '#404788',
              '#39568C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
              '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C667',
              '#73D055', '#95D840']
viridis_32 = np.flip(np.reshape(np.transpose(np.array([viridis_16, viridis_16])), [32,]))


offset = 0.5  # for 1 Hz
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
fig, ax = plt.subplots(1, 1, figsize=[12, 1.5])
for s in stim_times:
    ax.axvline(s, color=c_opto, alpha=0.5, linewidth=2)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
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

# plot just the average over all stim and save
for i in np.arange(len(data)):
    t_ts = data[i]['t_ts']
    bl_stim = data[i]['bl_stim']
    fig, ax = plt.subplots(1, 1, figsize=[4, 9])   
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
    for s in np.arange(data[i]['bl_stim'].segments[0].epochs[0].times.size):
        t = data[i]['bl_stim'].segments[0].epochs[0].times[s].rescale(pq.ms)
        d = data[i]['bl_stim'].segments[0].epochs[0].durations[s].rescale(pq.ms)
        ax.axvspan(t-t0, t+d-t0, ymin=0, ymax=1, color='c', alpha=0.25, zorder=1)
    ax.spines['left'].set_bounds(0, 100)
    ax.set_yticks([])
    ax.set_xlim([-5, 30])
    ax.set_xlabel('ms')
    ax.set_title(data_list['Filename'][i] + '\n(scalebar = 100 uV)')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_folder, data_list['Filename'][i] + '.png'), transparent=True)
    plt.close()


#%% figure: average trace from each frequency (opto_LFF only)

 
# plot just the average for each stim frequency and save
take_last = -15  # take only last stim to make average trace; set to 0 to keep all
for i in np.arange(len(data)):
    for f in np.arange(len(disp_freq)): 
        t_ts = data[i]['t_ts']
        bl_stim = data[i]['bl_'+disp_freq[f]]
        fig, ax = plt.subplots(1, 1, figsize=[4, 9])
        if len(bl_stim.segments) > 0:
            #plt.plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
            # make the average
            all_traces = np.array([s.analogsignals[0].magnitude for s in bl_stim.segments])
            try:
                grand_mean = np.mean(all_traces[take_last:], axis=0)*pq.uV    
            except ValueError:
                lengths = np.array([s.analogsignals[0].magnitude.shape[0] for s in bl_stim.segments])
                v, c = np.unique(lengths, return_counts=True)
                # don't include the ones that don't match
                idx = np.where(lengths == v[np.argmax(c)])[0]   
                grand_mean = np.mean(all_traces[idx][take_last:, :, :], axis=0)*pq.uV 
            for c in np.arange(nb_channel):
                plt.plot(t_ts, grand_mean[:, c]+offset_matrix[c], color=viridis_ordered[c])
            ax.spines['left'].set_bounds(0, 100)
        ax.set_xlim([-5, 30])
        ax.set_yticks([])
        ax.set_xlabel('ms')
        ax.set_title(data_list['Filename'][i] + '\n' + disp_freq[f] + '\n(scalebar = 100 uV)')
        fig.tight_layout()
        plt.savefig(os.path.join(fig_folder, 'avg_traces_'+disp_freq[f], data_list['Filename'][i] + '.png'), transparent=True)
        plt.close()


# for each cell, plot the average for each stim frequency, and overlay
disp_ch = np.arange(0, nb_channel, 4)
take_last = -15  # take only last stim to make average trace; set to 0 to keep all
for i in np.arange(len(data)):
    fig, ax = plt.subplots(1, len(disp_freq)+1, figsize=[12, 9], sharex=True, sharey=True,
                           constrained_layout=True)
    for f in np.arange(len(disp_freq)): 
        t_ts = data[i]['t_ts']
        bl_stim = data[i]['bl_'+disp_freq[f]]
        if len(bl_stim.segments) > 0:
            #ax[f].plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
            #ax[-1].plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
            # make the average
            all_traces = np.array([s.analogsignals[0].magnitude for s in bl_stim.segments])
            try:
                grand_mean = np.mean(all_traces[take_last:], axis=0)*pq.uV    
            except ValueError:
                lengths = np.array([s.analogsignals[0].magnitude.shape[0] for s in bl_stim.segments])
                v, c = np.unique(lengths, return_counts=True)
                # don't include the ones that don't match
                idx = np.where(lengths == v[np.argmax(c)])[0]   
                grand_mean = np.mean(all_traces[idx][take_last:, :, :], axis=0)*pq.uV 
            for c in np.arange(nb_channel):
                ax[f].plot(t_ts, grand_mean[:, c]+offset_matrix[c], color=viridis_ordered[c])
            for c in disp_ch:
                ax[-1].plot(t_ts, grand_mean[:, c]+offset_matrix[c], color=magma_4[f])
            ax[f].spines['left'].set_bounds(0, 100)
        ax[f].set_xlim([-5, 30])
        ax[f].set_yticks([])
        ax[f].set_xlabel('ms')
        ax[f].set_title(disp_freq[f], color=magma_4[f])
    ax[-1].spines['left'].set_bounds(0, 100)
    ax[-1].set_xlabel('ms')
    plt.suptitle(data_list['Filename'][i] + ', (scalebar = 100 uV)')
    plt.savefig(os.path.join(fig_folder, data_list['Filename'][i] + '.png'), transparent=True)
    plt.close()





#%% quantify fiber volley and fEPSP on average traces (opto_LFF only)
    

disp_ch = np.arange(0, nb_channel, 4)
take_last = -15  # take only last stim to make average trace; set to 0 to keep all
for i in np.arange(len(data)):
    all_freq_traces = []
    all_freq_mean = []
    all_ts = []
    for f in np.arange(len(disp_freq)):
        idx = np.array([0])
        bl_stim = data[i]['bl_'+disp_freq[f]]
        if len(bl_stim.segments) > 0:
            all_traces = np.array([s.analogsignals[0].magnitude for s in bl_stim.segments])
            try:
                grand_mean = np.mean(all_traces[take_last:], axis=0)*pq.uV    
            except ValueError:
                lengths = np.array([s.analogsignals[0].magnitude.shape[0] for s in bl_stim.segments])
                v, c = np.unique(lengths, return_counts=True)
                # don't include the ones that don't match
                idx = np.where(lengths == v[np.argmax(c)])[0]   
                grand_mean = np.mean(all_traces[idx][take_last:], axis=0)*pq.uV
            ts = bl_stim.segments[idx[0]].analogsignals[0].times
            ts = ts -ts[0] + e_win[0]
        all_freq_traces.append(all_traces)
        all_freq_mean.append(grand_mean)
        all_ts.append(ts.rescale(pq.ms))
    data[i]['traces'] = all_freq_traces
    data[i]['mean_traces'] = all_freq_mean
    data[i]['traces_ts'] = all_ts


# find the channel with the biggest downward fEPSP in each frequency
sl_channel = np.full((len(data), len(disp_freq)), np.nan)
for i in np.arange(len(data)):
    for f in np.arange(len(disp_freq)):
        ts = data[i]['traces_ts'][f]
        mean_traces = data[i]['mean_traces'][f]
        idx0 = np.searchsorted(ts, 0)
        idx1 = np.searchsorted(ts, 6)
        idx2 = np.searchsorted(ts, 20)
        fEPSP_amp = np.min(mean_traces[idx1:idx2, :], axis=0) - mean_traces[idx0, :]
        sl_channel[i, f] = np.argmin(fEPSP_amp)
sl_channel = sl_channel.astype(int)

has_spikes = np.full((len(data), len(disp_freq)), np.nan)
for i in np.arange(len(data)):
    for f in np.arange(len(disp_freq)):
        has_spikes[i, f] = data[i]['bl'].segments[0].analogsignals[0].array_annotations['has_spikes'][sl_channel[i, f]]

## for each recording, plot all the channel means in grey, and the seleced one in black
#for i in np.arange(len(data)):
#    plt.figure()
#    plt.plot(data[i]['traces_ts'][3], data[i]['mean_traces'][3], color=c_mgry)
#    plt.plot(data[i]['traces_ts'][3], data[i]['mean_traces'][3][:, sl_channel[i, 3]], color=c_blk)

# calculate the fEPSP amplitude for each channel, frequency, and recording
data_list['selected channel'] = sl_channel[:, 3]
for f in np.arange(len(disp_freq)):
    fEPSPs = np.full(len(data), np.nan)
    for i in np.arange(len(data)):
        ts = data[i]['traces_ts'][f]
        mean_traces = data[i]['mean_traces'][f]
        idx0 = np.searchsorted(ts, 0)
        idx1 = np.searchsorted(ts, 6)
        idx2 = np.searchsorted(ts, 20)
        fEPSP_amp = np.min(mean_traces[idx1:idx2, :], axis=0) - mean_traces[idx0, :]
        fEPSPs[i] = fEPSP_amp[sl_channel[i, 3]]
    data_list['fEPSP_amp_'+disp_freq[f]] = fEPSPs

# plot the percent change in fEPSP amp from 0.1 Hz
fig, ax = plt.subplots(1, 1)
for f in np.arange(3):
    ax.scatter(f*np.ones(len(data))+0.5*(data_list['Cre']=='T'),
               100*data_list['fEPSP_amp_'+disp_freq[f+1]]/data_list['fEPSP_amp_0.1 Hz'],
               color=magma_4[f+1])
ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
ax.set_xticklabels(['0.5 Hz\nctl', '0.5 Hz\ncre', '1 Hz\nctl', '1 Hz\ncre', '3 Hz\nctl', '3 Hz\ncre'])
ax.set_ylim([0, 500])
ax.set_ylabel('fEPSP amplitude (% change from 0.1 Hz)')

## plot only those sessions where stim is 3.5 power and 5 ms
#stim_dur = np.round(np.array([s['bl'].segments[0].epochs[0].durations[0] for s in data]), decimals=3)
#data_list_filt = data_list[np.logical_and(data_list['Laser Power'] == 3.5, stim_dur == 0.005)]    
#fig, ax = plt.subplots(1, 1)
#for f in np.arange(3):
#    ax.scatter(f*np.ones(len(data_list_filt))+0.5*(data_list_filt['Cre']=='T'),
#               100*data_list_filt['fEPSP_amp_'+disp_freq[f+1]]/data_list_filt['fEPSP_amp_0.1 Hz'])

# plot the mean traces and mark the fEPSP peak
fig, ax = plt.subplots(4, 6, figsize=[16, 10], sharex=True, sharey=True)
axs = ax.flat
for i in np.arange(len(data)):
    for f in np.arange(len(disp_freq)):
        c = data_list['selected channel'][i]
        ts = data[i]['traces_ts'][f]
        trace = data[i]['mean_traces'][f][:, c]
        axs[i].plot(ts, trace, color=magma_4[f])
        idx1 = np.searchsorted(ts, 6)
        idx2 = np.searchsorted(ts, 20)
        idx = np.argmin(trace[idx1:idx2])
        axs[i].scatter(ts[idx+idx1], trace[idx+idx1], color=magma_4[f])
        axs[i].set_title(data_list['Filename'][i] + ', ch' + str(data_list['selected channel'][i]), fontsize=10)
axs[0].set_xlim([-10, 30])
axs[18].set_xlabel('ms')
axs[18].set_ylabel('uV')

# plot the mean traces and mark the fEPSP peak
# version: just one example recording
i = 8
fig, axs = plt.subplots(1, 1, figsize=[4, 4])
t0 = data[i]['bl_stim'].segments[0].analogsignals[0].times[0].rescale(pq.ms)
t0 = t0-e_win[0]
for s in np.arange(data[i]['bl_stim'].segments[0].epochs[0].times.size):
    t = data[i]['bl_stim'].segments[0].epochs[0].times[s].rescale(pq.ms)
    d = data[i]['bl_stim'].segments[0].epochs[0].durations[s].rescale(pq.ms)
    axs.axvspan(t-t0, t+d-t0, ymin=0, ymax=1, color='c', alpha=0.1, zorder=1)
for f in np.arange(len(disp_freq)):
    #c = data_list['selected channel'][i]
    c = sl_channel[i, -1]
    ts = data[i]['traces_ts'][f]
    trace = data[i]['mean_traces'][f][:, c]
    axs.plot(ts, trace, color=magma_4[f])
    idx1 = np.searchsorted(ts, 6)
    idx2 = np.searchsorted(ts, 20)
    idx = np.argmin(trace[idx1:idx2])
    axs.scatter(ts[idx+idx1], trace[idx+idx1], color=magma_4[f])
    idx0 = np.searchsorted(ts, 0)
    axs.scatter(ts[idx0], trace[idx0], color=magma_4[f])
axs.set_title(data_list['Filename'][i] + ', ch' + str(c), fontsize=10)
axs.set_xlim([-10, 30])
axs.spines['left'].set_bounds(-250, -750)
axs.set_yticks([])
axs.set_xticks([-10, 0, 10, 20, 30])
axs.set_xticklabels(['', 0, '', 20, ''])
axs.set_xlabel('ms')
fig.tight_layout()


# %% find the fEPSP amplitude for each stim (on only the selected channel)

fEPSP_amp = np.full((len(data), len(data[0]['bl_stim'].segments)), np.nan)
for i in np.arange(len(data)):
    for s in np.arange(len(data[i]['bl_stim'].segments)):
        c = data_list['selected channel'][i]
        trace = data[i]['bl_stim'].segments[s].analogsignals[0].magnitude[:, c]
        ts = data[i]['bl_stim'].segments[s].analogsignals[0].times.rescale(pq.ms)
        ts = ts -ts[0] + e_win[0]
        idx0 = np.searchsorted(ts, 0)
        idx1 = np.searchsorted(ts, 6)
        idx2 = np.searchsorted(ts, 20)
        fEPSP_amp[i, s] = np.min(trace[idx1:idx2], axis=0) - trace[idx0]

# plot % change over stim (baseline is first 5 stim at 0.1 Hz)
fEPSP_amp_perc = 100*(fEPSP_amp/np.atleast_2d(np.mean(fEPSP_amp[:, :5], axis=1)).T)
fig, ax = plt.subplots(1, 1)
for i in np.arange(len(data)):
    if data_list['Cre'][i] == 'T':
        color='b'
    else:
        color='k'
    ax.scatter(np.arange(fEPSP_amp.shape[1]), fEPSP_amp_perc[i, :], color=color)

# plot % change over stim, avg+/- std
cre_perc = fEPSP_amp_perc[data_list['Cre'] == 'T', :]
ctrl_perc = fEPSP_amp_perc[data_list['Cre'] == 'F', :]
fig, ax = plt.subplots(1, 1)
ax.axhline(100, color=c_blk, zorder=1)
ax.axvspan(20, 50, ymin=0, ymax=1, color=magma_4[1], alpha=0.25, zorder=1)
ax.axvspan(70, 100, ymin=0, ymax=1, color=magma_4[2], alpha=0.25, zorder=1)
ax.axvspan(120, 150, ymin=0, ymax=1, color=magma_4[3], alpha=0.25, zorder=1)
ax.scatter(np.arange(fEPSP_amp.shape[1]), np.nanmedian(ctrl_perc, axis=0),
           color=c_blk, zorder=2)
ax.scatter(np.arange(fEPSP_amp.shape[1]), np.nanmedian(cre_perc, axis=0),
           color=c_mgry, zorder=3)
ax.set_title('all sessions, median (gray=Cre)')
ax.set_ylabel('fEPSP amplitude (% change from 0.1 Hz)')
ax.set_xlabel('stim number')
plt.tight_layout()




