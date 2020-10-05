# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:48:07 2020

@author: akees
"""

# GluK2 anesthetized optogenetic stimulation experiments
# opto stim in DG, record in CA3
# protocol: opto_LFF


#%% import modules

import neo  
from neo import io
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



# %% definitions

# definition for downsampling
def ds(ts, trace, ds_factor):
    signal_ds = np.mean(np.resize(trace,
                        (int(np.floor(trace.size/ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor/2)), ts.size, ds_factor)]
    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds

# load intan data for the opto_LFF protocol
def load_data(rec_idx, data_list, order_type='custom', highpass=0.2, lowpass=300):
    
    # build the path
    root = r'Z:\Data\GluK2_in_vivo'
    date = str(data_list['Date'][rec_idx].date())
    session = data_list['Filename'][rec_idx]
    data_folder = os.path.join(root, date, session)
    
    # use intan's provided reader for reading the header
    info_file = os.path.join(data_folder, 'info.rhd')
    header = read_data(info_file)
    fs = header['frequency_parameters']['amplifier_sample_rate']
    nb_channel = len(header['amplifier_channels'])
    nb_digin = len(header['board_dig_in_channels'])
    fs_digin = header['frequency_parameters']['board_dig_in_sample_rate']*pq.Hz

    # use neo to read the amplifier file
    amplifier_file = 'amplifier.dat'
    file_name = os.path.join(data_folder, amplifier_file)
    r = io.RawBinarySignalIO(file_name, dtype='int16', sampling_rate=fs,
                             nb_channel=nb_channel, signal_gain=0.195)
    bl = r.read_block(signal_group_mode='group-by-same-units')
    # include the appropraite units
    bl.segments[0].analogsignals[0] = bl.segments[0].analogsignals[0]*pq.uV
    # change the name
    bl.segments[0].analogsignals[0].name = 'amplifier'
    
    # if you want to put it in depth order right away
    if order_type == 'custom':
        depth_order = np.array([d['custom_order'] for d in header['amplifier_channels']], dtype=int)
        bl.segments[0].analogsignals[0] = bl.segments[0].analogsignals[0][:, np.argsort(depth_order)]
        bl.annotate(order_type = 'custom')
        amplifier_channels = np.array(header['amplifier_channels'])[np.argsort(depth_order)]
    else:  # keep the native order if there is no input or unrecognized
        depth_order = np.array([d['custom_order'] for d in header['amplifier_channels']], dtype=int)
        bl.annotate(depth_order = depth_order)
        bl.annotate(order_type = 'native')
        amplifier_channels = np.array(header['amplifier_channels'])
    
    # the time.dat file is not necessary; starts at 0 and gives intergers for each
    # timestamp which then need to be divided by the sampling rate
    # neo IO does this anyway using the indices of the sampling points
    
    # add the digital in signals to the segment
    # this file is interesting - the 16 possible digital inputs are saved in the
    # same bit word
    digin_file = 'digitalin.dat'
    file_name = os.path.join(data_folder, digin_file)
    digital_word = np.fromfile(file_name, np.uint16)
    # change dtype to float
    digital_word = np.array(digital_word, dtype='float32')
    # TODO: when there's more than one channel, the values add - need to figure out
    # how to separate them (i.e. if ch0 and ch1 are True at the same time,
    # digital_word = 2^0 + 2^1 = 3
    name = 'digital in'
    array_annotations = {'wavelength':450*pq.nm, 'OF_diameter':200*pq.um}
    anasig = neo.AnalogSignal(digital_word, units=pq.dimensionless, sampling_rate=fs_digin,
                              name=name, file_origin=file_name,
                              array_annotations=array_annotations)
    bl.segments[0].analogsignals.append(anasig)
    
    # turn the digital inputs into epoch
    on = bl.segments[0].analogsignals[0].times[np.where(np.ediff1d(1*digital_word) > 0)[0]]
    off = bl.segments[0].analogsignals[0].times[np.where(np.ediff1d(1*digital_word) < 0)[0]]
    # find the pattern of stimulation, and label each stimulation
    isi = np.round(np.diff(on).rescale(pq.ms))
    isi_values, isi_index, isi_counts = np.unique(isi, return_index=True, return_counts=True)
    stim_name = data_list['Axon Protocol'][rec_idx]
    # opto_LFF
    if stim_name == 'opto_LFF':
        stim_freq = np.round((1/isi).rescale(pq.Hz), decimals=1)
        labels = np.full(on.size, np.nan, dtype=object)
        for i in np.arange(on.size-1):
            labels[i+1] = str(stim_freq[i])
        labels[0] = labels[1]
        labels = labels.astype('S')
    # one frequency
    elif isi_index.size == 1:
        stim_freq = np.round((1/isi_values[0]).rescale(pq.Hz), decimals=1)
        stim_name = str(stim_freq)
        labels = np.arange(on.size).astype('S') # not sure if this is the good format
    # train stim
    else:
        stim_freq = np.round((1/isi_values[0]).rescale(pq.Hz), decimals=0)
        num_stim = isi_index[1] + 1
        num_train = isi_counts[1]+1
        stim_name = str(num_stim) + 'x' + str(stim_freq)
        train_labels = np.repeat(np.arange(num_train), num_stim).astype('S')
        train_labels = train_labels[0:on.size]
        stim_labels = np.tile(np.arange(num_stim), num_train).astype('S')
        stim_labels = train_labels[0:on.size]
        labels = np.char.add(train_labels, np.full(on.size, '_').astype('S'))
        labels = np.char.add(labels, stim_labels)
         
    epoch = neo.Epoch(times=on, durations=off-on, labels=labels, name=stim_name)
    bl.segments[0].epochs.append(epoch)
    bl.annotate(dig_in_channels = [d['native_channel_name'] for d in header['board_dig_in_channels']])
    
    # when applicable, add another epoch object that is the train rather than the individual pulses
    # opto_LFF
    if isi_index.size > 2:
        train_start = on[np.ediff1d(isi, to_begin=0, to_end=0) != 0]
        train_start = np.append(on[0], train_start)*on.units
        train_stop = train_start[1:]
        train_stop = np.append(train_stop, off[-1])*on.units
        stim_freqs = stim_freq[np.searchsorted(on, train_start)]
        train_labels = np.full(train_start.size, np.nan, dtype=object)
        for i in np.arange(train_start.size):
            train_labels[i] = str(stim_freqs[i])
        train_labels = train_labels.astype('S')
        train_name = stim_name + ' train'
        epoch_trains = neo.Epoch(times=train_start, durations=train_stop-train_start,
                                 labels=train_labels, name=train_name)
        bl.segments[0].epochs.append(epoch_trains)
    # train stim
    elif isi_index.size == 2:
        train_start = on[np.arange(0, on.size, num_stim)]
        train_stop = off[np.arange(num_stim-1, on.size, num_stim)]
        train_labels = np.arange(0, train_start.size).astype('S')
        train_name = stim_name + ' train'
        epoch_trains = neo.Epoch(times=train_start, durations=train_stop-train_start,
                                 labels=train_labels, name=train_name)
        bl.segments[0].epochs.append(epoch_trains)
    
    
    # Oliva Buzsaki 2016:
    # CA pyr layers determined by presence of spikes (>500 Hz) and ripples
    # used CSD to determine dendritic layers (Fernández-Ruiz 2012, 2013)
    # for now, do a quick and dirty detection of channels that have spikes
    trace = bl.segments[0].analogsignals[0].magnitude
    # design highpass filter for spikes
    nyq = fs/2
    b_sp, a_sp = signal.butter(4, 500/nyq, "high", analog=False)
    has_spikes = np.zeros(nb_channel)
    occ_spikes = np.zeros(nb_channel)
    for c in np.arange(nb_channel):
        #trace_sp = np.abs(stats.zscore(signal.filtfilt(b_sp, a_sp, trace[:, c])))
        trace_sp = np.abs(stats.zscore(signal.filtfilt(b_sp, a_sp, trace[:, c])))
        if np.any(trace_sp > 8):
            has_spikes[c] = 1
            occ_spikes[c] = np.log10(np.sum(trace_sp > 8)/trace.shape[0])
    
    
    # for each channel: filter out spikes and
    # downsample/average to a final frequency of 1250 Hz
    # makes no difference whether you downsample or filter first
    # beware that there are spike artifacts in the trace
    ds_factor = 16
    trace = bl.segments[0].analogsignals[0].magnitude
    ts = bl.segments[0].analogsignals[0].times
    trace_ds_filt = np.full((int(np.floor(ts.size/ds_factor)), nb_channel), np.nan)
    # design highpass and lowpass filters
    nyq = fs/(2*ds_factor)
    b_hi, a_hi = signal.butter(4, highpass/nyq, "high", analog=False)
    b_lo, a_lo = signal.butter(4, lowpass/nyq, "low", analog=False)
    for c in np.arange(nb_channel):
        ts_ds, trace_ds = ds(ts, trace[:, c], ds_factor)
        trace_highpass = signal.filtfilt(b_hi, a_hi, trace_ds)
        trace_ds_filt[:, c] = signal.filtfilt(b_lo, a_lo, trace_highpass)
        
    name = 'filtered'
    array_annotations = {'has_spikes':has_spikes, 'occ_spikes':occ_spikes,
                         'native_channel':[d['native_channel_name'] for d in amplifier_channels],
                         'custom_channel':[d['custom_order'] for d in amplifier_channels],
                         'impedance_magnitude':[d['electrode_impedance_magnitude'] for d in amplifier_channels],
                         'impedance_phase':[d['electrode_impedance_phase'] for d in amplifier_channels]}
    annotations = {'highpass':highpass*pq.Hz, 'lowpass':lowpass*pq.Hz,
                   'filter_type':'butter', 'filter_order':4, 'ds_factor':ds_factor,
                   'order_type':order_type}
    anasig = neo.AnalogSignal(trace_ds_filt, units=pq.uV, sampling_rate=(fs/ds_factor)*pq.Hz,
                              name=name, file_origin=file_name, annotations=annotations,
                              array_annotations=array_annotations)
    bl.segments[0].analogsignals.append(anasig)
      
    return bl
   
   

     
    
#%% load data

# load the excel sheet list of data
sheet_name = 'opto_10x20Hz'
data_list = pd.read_excel(r"C:\Users\akees\Dropbox\GluK2\viral_injections_GluK2_DG_in_vivo.xlsx",
                          sheet_name=sheet_name)

# create the list of data dictionaries
data = [{} for i in np.arange(data_list.shape[0])]

#for i in np.arange(len(data)):
for i in tqdm(np.arange(len(data))):
    
    # load the intan file
    bl = load_data(i, data_list, order_type='custom')
    
    data[i]['bl'] = bl




#%% set number of channels (assume same for all sessions)
    
nb_channel = data[0]['bl'].segments[0].analogsignals[0].magnitude.shape[1]

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
fig_folder = r'C:\Users\akees\Documents\Ashley\Analysis\GluK2\anesthetized\opto_stim\opto_LFF'


# %% figure - raw traces and check filter settings

start = 140000
ds_idx = np.arange(start, start + 20000)
plt.figure()
for c in np.arange(nb_channel):
    plt.plot(bl.segments[0].analogsignals[0].times[ds_idx],
         (bl.segments[0].analogsignals[0]+offset_matrix[c]).magnitude[ds_idx, c],
         color=viridis_ordered[c])
plt.plot(bl.segments[0].analogsignals[0].times[ds_idx],
         np.max(offset_matrix)*pq.mV.rescale(pq.uV)*bl.segments[0].analogsignals[1].magnitude[ds_idx], color=c_blk)
plt.tight_layout() 

start = int(np.round(140000/16))
ds_idx_lf = np.arange(start, start + int(np.round(20000/16)))
plt.figure()
for c in np.arange(nb_channel):
    plt.plot(bl.segments[0].analogsignals[2].times[ds_idx_lf],
         (bl.segments[0].analogsignals[2]+offset_matrix[c]).magnitude[ds_idx_lf, c],
         color=viridis_ordered[c])
plt.plot(bl.segments[0].analogsignals[0].times[ds_idx],
         np.max(offset_matrix)*pq.mV.rescale(pq.uV)*bl.segments[0].analogsignals[1].magnitude[ds_idx], color=c_blk)
plt.tight_layout() 


#%% figure: raw traces

# plot the raw traces for a session (downsample for visualization)
i = 0
ds_idx = np.arange(0, data[i]['bl'].segments[0].analogsignals[0].shape[0], 50)
ds_idx = np.arange(0, 20000)
plt.figure()
for c in np.arange(data[i]['bl'].segments[0].analogsignals[0].shape[1]):
    plt.plot(data[i]['bl'].segments[0].analogsignals[0].times[ds_idx],
         (data[i]['bl'].segments[0].analogsignals[0]+offset_matrix[c]).magnitude[ds_idx, c],
         color=viridis_ordered[c])
plt.plot(data[i]['bl'].segments[0].analogsignals[0].times[ds_idx],
         16000*data[i]['bl'].segments[0].analogsignals[1].magnitude[ds_idx], color=c_blk)
plt.tight_layout()


#%% figure: average trace over all stim (every frequency)

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


# plot just the average over all stim and save
for i in np.arange(len(data)):
    t_ts = data[i]['t_ts']
    bl_stim = data[i]['bl_stim']
    fig, ax = plt.subplots(1, 1, figsize=[4, 9])
    plt.plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
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
        plt.plot(t_ts, grand_mean[:, c]+offset_matrix[c], color=viridis_ordered[c])
    ax.set_xlim([-5, 30])
    ax.spines['left'].set_bounds(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('ms')
    ax.set_title(data_list['Filename'][i] + '\n(scalebar = 100 uV)')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_folder, data_list['Filename'][i] + '.png'), transparent=True)
    plt.close()


#%% figure: average trace from each frequency (opto_LFF)


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
    
    
# plot just the average for each stim frequency and save
disp_freq = ['0.1 Hz', '0.5 Hz', '1.0 Hz', '3.0 Hz']
take_last = -15  # take only last stim to make average trace; set to 0 to keep all
for i in np.arange(len(data)):
    for f in np.arange(len(disp_freq)): 
        t_ts = data[i]['t_ts']
        bl_stim = data[i]['bl_'+disp_freq[f]]
        fig, ax = plt.subplots(1, 1, figsize=[4, 9])
        if len(bl_stim.segments) > 0:
            plt.plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
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
disp_freq = ['0.1 Hz', '0.5 Hz', '1.0 Hz', '3.0 Hz']
disp_ch = np.arange(0, nb_channel, 4)
take_last = -15  # take only last stim to make average trace; set to 0 to keep all
for i in np.arange(len(data)):
    fig, ax = plt.subplots(1, len(disp_freq)+1, figsize=[12, 9], sharex=True, sharey=True,
                           constrained_layout=True)
    for f in np.arange(len(disp_freq)): 
        t_ts = data[i]['t_ts']
        bl_stim = data[i]['bl_'+disp_freq[f]]
        if len(bl_stim.segments) > 0:
            ax[f].plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
            ax[-1].plot(t_ts, np.max(offset_matrix).rescale(pq.uV)*bl_stim.segments[0].analogsignals[1].magnitude, color=c_mgry)
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



#%% quantify fiber volley and fEPSP on average traces
    
    
disp_freq = ['0.1 Hz', '0.5 Hz', '1.0 Hz', '3.0 Hz']
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
            all_traces = np.array([s.analogsignals[2].magnitude for s in bl_stim.segments])
            try:
                grand_mean = np.mean(all_traces[take_last:], axis=0)*pq.uV    
            except ValueError:
                lengths = np.array([s.analogsignals[2].magnitude.shape[0] for s in bl_stim.segments])
                v, c = np.unique(lengths, return_counts=True)
                # don't include the ones that don't match
                idx = np.where(lengths == v[np.argmax(c)])[0]   
                grand_mean = np.mean(all_traces[idx][take_last:], axis=0)*pq.uV
            ts = bl_stim.segments[idx[0]].analogsignals[2].times
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
        has_spikes[i, f] = data[i]['bl'].annotations['has_spikes'][sl_channel[i, f]]

# for each recording, plot all the channel means in grey, and the seleced one in blue
for i in np.arange(len(data)):
    plt.figure()
    plt.plot(data[i]['traces_ts'][3], data[i]['mean_traces'][3], color=c_mgry)
    plt.plot(data[i]['traces_ts'][3], data[i]['mean_traces'][3][:, sl_channel[i, 3]], color=c_blk)

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

# plot only those sessions where stim is 3.5 power and 5 ms
stim_dur = np.round(np.array([s['bl'].segments[0].epochs[0].durations[0] for s in data]), decimals=3)
data_list_filt = data_list[np.logical_and(data_list['Laser Power'] == 3.5, stim_dur == 0.005)]    

fig, ax = plt.subplots(1, 1)
for f in np.arange(3):
    ax.scatter(f*np.ones(len(data_list_filt))+0.5*(data_list_filt['Cre']=='T'),
               100*data_list_filt['fEPSP_amp_'+disp_freq[f+1]]/data_list_filt['fEPSP_amp_0.1 Hz'])

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




