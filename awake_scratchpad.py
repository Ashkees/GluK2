# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:58:53 2020

@author: akees
"""

# GluK2 awake experiments (no task)
# record in CA3/DG
# load raw data, filter and downsample to 1250 Hz, parse digital input,
# synchronize with wheel, save as .nix


#%% import modules

import neo  
from neo import io
from neo import NixIO
import numpy as np
import os
import quantities as pq
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm
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

# load intan data for awake recordings with wheel
def load_data(rec_idx, data_list, order_type='custom', highpass=0.2, lowpass=300):
    
    # build the path
    root = r'Z:\Data\GluK2_in_vivo'
    date = str(data_list['Date'][rec_idx].date())
    session = data_list['Ephy Filename'][rec_idx]
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
    # desired organization: by shank, then by depth
    # TODO: not in the right order WTF
    if order_type == 'custom':
        # first, determine which type of ordering was used in the config
        if data_list['Probe'][rec_idx] in ['S404', 'S406', 'Q631']:
            num_shank = 4
            if header['amplifier_channels'][60]['custom_order'] == 1:
                depth_order = np.array([d['custom_order'] for d in header['amplifier_channels']], dtype=int)
                shank_order = np.empty(0)
                for s in np.arange(num_shank):
                    ind = np.arange(0+s, nb_channel, num_shank)
                    shank_order = np.append(shank_order, np.argsort(depth_order)[ind])
                shank_order = shank_order.astype(int)
                bl.segments[0].analogsignals[0] = bl.segments[0].analogsignals[0][:, shank_order]
                bl.annotate(order_type = 'shank')
                amplifier_channels = np.array(header['amplifier_channels'])[shank_order]
            elif header['amplifier_channels'][60]['custom_order'] == 16:
                depth_order = np.array([d['custom_order'] for d in header['amplifier_channels']], dtype=int)
                shank_order = np.argsort(depth_order)
                bl.segments[0].analogsignals[0] = bl.segments[0].analogsignals[0][:, shank_order]
                bl.annotate(order_type = 'shank')
                amplifier_channels = np.array(header['amplifier_channels'])[shank_order]
            else:
                print('uninterpretable channel order - keeping native order')
                bl.annotate(order_type = 'native')
                amplifier_channels = np.array(header['amplifier_channels'])
        else:
            print('uninterpretable channel order - keeping native order')
            bl.annotate(order_type = 'native')
            amplifier_channels = np.array(header['amplifier_channels'])
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
    isi = np.round(np.diff(on).rescale(pq.ms), decimals=-1)
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
        stim_labels = stim_labels[0:on.size]
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
sheet_name = 'awake'
output_folder = r'C:\Users\akees\Documents\Ashley\Datasets\GluK2\awake'


data_list = pd.read_excel(r"C:\Users\akees\Documents\Ashley\Analysis\GluK2\Data_to_Analyze_GluK2.xlsx",
                          sheet_name=sheet_name)


#for i in np.arange(len(data)):
for i in tqdm(np.arange(len(data_list))):
    
    # load the intan file
    bl = load_data(i, data_list, order_type='custom')
    
    # delete any analogsignals with a sampling rate higher than 8 kHz
    for s in np.arange(len(bl.segments)):
        sampling_rate = np.array([s.sampling_rate for s in bl.segments[s].analogsignals])
        while np.any(sampling_rate > 8000*pq.Hz):
            to_delete = np.where(sampling_rate > 8000*pq.Hz)[0][0]
            del bl.segments[s].analogsignals[to_delete]
            sampling_rate = np.array([s.sampling_rate for s in bl.segments[s].analogsignals])
    
    # save the filtered traces as .nix files
    output_file = os.path.join(output_folder, data_list['Filename'][i] + '.nix')
    nio = NixIO(output_file, 'ow')
    nio.write_block(bl)
    nio.close()
    
    

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
fig_folder = os.path.join(r'C:\Users\akees\Documents\Ashley\Analysis\GluK2\anesthetized\opto_stim',
                          sheet_name)


# %% figure - raw traces

start = 1000
ds_idx = np.arange(start, start + 20000)
plt.figure()
for c in np.arange(16):
    plt.plot(bl.segments[0].analogsignals[0].times[ds_idx],
         (bl.segments[0].analogsignals[0]+offset_matrix[c]).magnitude[ds_idx, c],
         color=viridis_ordered[c])
plt.tight_layout() 

    


