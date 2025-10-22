# finding Cdinv for real stations
# importing packages !
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from tqdm import tqdm
import corner
import time
from collections import Counter
from matplotlib.ticker import MaxNLocator
import pickle as pickle
from scipy.optimize import minimize
import obspy
import pandas as pd
from datetime import datetime
from obspy.geodetics import gps2dist_azimuth
from obspy import Stream
from obspy.signal.rotate import rotate_ne_rt

from obspy.clients.syngine import Client
client = Client()
from obspy.clients.fdsn import Client
ap = Client('IRIS')

from pyTransC import TransC_Sampler

###################################

### information needed as input

# station information - inventory, network, station, latitude, longitude
# event information - latitude, longitude, time
# splitting information - slice duration, overlap, filter frequencies (max and min), noise max % 

# calculate the distance between two stations
def distance(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0]
def back_azimuth(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[2]

def acquire_waveform(network, station, inv, start_time, end_time):
    waveform_z = ap.get_waveforms(network=network, station=station, location='*', channel='LHZ', starttime=start_time, endtime=end_time)
    waveform_z.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.1)
    waveform_z.detrend('linear')
    waveform_z.detrend('demean')
    waveform_z.taper(max_percentage=0.1)

    waveform_n = ap.get_waveforms(network=network, station=station, location='*', channel='LHN', starttime=start_time, endtime=end_time)
    waveform_n.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.1)
    waveform_n.detrend('linear')
    waveform_n.detrend('demean')
    waveform_n.taper(max_percentage=0.1)

    waveform_e = ap.get_waveforms(network=network, station=station, location='*', channel='LHE', starttime=start_time, endtime=end_time)
    waveform_e.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.1)
    waveform_e.detrend('linear')
    waveform_e.detrend('demean')
    waveform_e.taper(max_percentage=0.1)
    
    return waveform_z, waveform_n, waveform_e

def split_waveform_overlapping(stream, slice_duration=800, overlap=400):
    """
    Split a waveform into overlapping segments.
    
    Parameters:
    stream: ObsPy Stream object
    slice_duration: Duration of each slice in seconds (default 800)
    overlap: Overlap between adjacent slices in seconds (default 400)
    
    Returns:
    List of ObsPy Stream objects (slices)
    """
    slices = Stream()
    
    for tr in stream:
        # Calculate step size (slice_duration - overlap)
        step = slice_duration - overlap
        
        # Get trace duration
        trace_duration = tr.stats.npts / tr.stats.sampling_rate
        
        # Calculate number of slices
        num_slices = int((trace_duration - overlap) / step)
        
        start_time = tr.stats.starttime
        
        for i in range(num_slices):
            # Calculate slice start and end times
            slice_start = start_time + (i * step)
            slice_end = slice_start + slice_duration
            
            # Create slice
            slice_stream = stream.copy()
            slice_stream.trim(slice_start, slice_end)
            
            # Only add if slice has data
            if len(slice_stream) > 0 and slice_stream[0].stats.npts > 0:
                slices += slice_stream
                #print(f"Slice {i+1}: {slice_start} to {slice_end} ({slice_stream[0].stats.npts} samples)")
    
    return slices

def filter_and_trim(waveform_slices, freqmin=0.04, freqmax=0.06):
    for i in range(len(waveform_slices)):
        waveform_slices[i].filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    for i in range(len(waveform_slices)):
        waveform_slices[i].trim(starttime=waveform_slices[i].stats.starttime + 200, endtime=waveform_slices[i].stats.endtime - 200)
    return waveform_slices

def noise_filterer(waveform_slices, event_waveform, expected_length=400, noise_max=15):
    low_amplitude_slices = Stream()
    low_amplitude_data = []

    max_amplitude = max(abs(event_waveform[0].data))
    max_amplitude_percent = max_amplitude * noise_max / 100

    for trace in waveform_slices:
        if max(abs(trace.data)) < max_amplitude_percent:
            # Check if trace data is longer than 400 samples
            if len(trace.data) > expected_length:
                # Trim to first expected_length samples
                trace.data = trace.data[:expected_length]
                # Update trace stats to reflect new number of points
                trace.stats.npts = expected_length
            elif len(trace.data) < expected_length:
                # Skip traces that are too short
                print(f"Skipping trace with only {len(trace.data)} samples (need {expected_length})")
                continue

            # Add trace if it has exactly expected_length samples
            if len(trace.data) == expected_length:
                low_amplitude_slices += trace
                low_amplitude_data.append(trace.data)

    print(f"Selected {len(low_amplitude_slices)} out of {len(waveform_slices)} traces with amplitude < {max_amplitude_percent:.2e} and exactly {expected_length} samples")

    return low_amplitude_slices, low_amplitude_data


def plot_noisy_data(low_amplitude_slices_z, low_amplitude_slices_n, low_amplitude_slices_e, components='RT'):
    fig = plt.figure(figsize=(24, 6))
    axes = []

    ax_z = plt.subplot(1, 3, 1)
    ax_n = plt.subplot(1, 3, 2)
    ax_e = plt.subplot(1, 3, 3)


    for i in range(len(low_amplitude_slices_z)):
        ax_z.scatter(low_amplitude_slices_z[i].times(), low_amplitude_slices_z[i].data + i*1e-6, s=1, color='black')
    ax_z.set_title('Z component noisy data')

    for i in range(len(low_amplitude_slices_n)):
        ax_n.scatter(low_amplitude_slices_n[i].times(), low_amplitude_slices_n[i].data + i*1e-6, s=1, color='black')

    for i in range(len(low_amplitude_slices_e)):
        ax_e.scatter(low_amplitude_slices_e[i].times(), low_amplitude_slices_e[i].data + i*1e-6, s=1, color='black')

    if components == 'NE':
        ax_n.set_title('N component noisy data')
        ax_e.set_title('E component noisy data')

    elif components == 'RT':
        ax_n.set_title('R component noisy data')
        ax_e.set_title('T component noisy data')
        
    axes.append(ax_z)
    axes.append(ax_n)
    axes.append(ax_e)

    plt.show()
    return axes


# slice duration, overlap, filter frequencies (max and min), noise max %
# station information - inventory, network, station, latitude, longitude
# event information - latitude, longitude, time
six_hours = 60*60*6
half_hour = 60*30

def real_station_noise_samples(station_info, event_info, slice_duration=800, overlap=400, freqmin=0.04, freqmax=0.06, sampling_length=60*60*6, sampling_start=60*30, noise_max_z=15, noise_max_r=15, noise_max_t=15, noise_max_n=15, noise_max_e=15):
    '''
    ouputs data z, data r, data t, data n, data e
    '''
    inv, network, station, sta_lat, sta_lon = station_info
    event_lat, event_lon, event_time = event_info
    
    # downloading 6 hours of potentially noisy data
    event_station_distance = distance(event_lat, event_lon, sta_lat, sta_lon)/1000 # in km
    event_time_to_arrive   = event_station_distance / 2.95 # 2.95 km/s wavespeed

    # six_hours = 60*60*6
    # half_hour = 60*30
    
    six_hours = sampling_length
    half_hour = sampling_start

    begin_time = obspy.UTCDateTime(event_time) + event_time_to_arrive - six_hours - half_hour # 6:30 before the event
    end_time   = obspy.UTCDateTime(event_time) + event_time_to_arrive - half_hour # 30 minutes before the event
    
    waveform_z, waveform_n, waveform_e = acquire_waveform(network, station, inv, begin_time, end_time)

    # splitting the waveform into overlapping segments
    waveform_slices_z = split_waveform_overlapping(waveform_z, slice_duration=slice_duration, overlap=overlap)
    waveform_slices_n = split_waveform_overlapping(waveform_n, slice_duration=slice_duration, overlap=overlap)
    waveform_slices_e = split_waveform_overlapping(waveform_e, slice_duration=slice_duration, overlap=overlap)
    
    # filtering and trimming the waveform slices
    waveform_slices_z = filter_and_trim(waveform_slices_z, freqmin=freqmin, freqmax=freqmax)
    waveform_slices_n = filter_and_trim(waveform_slices_n, freqmin=freqmin, freqmax=freqmax)
    waveform_slices_e = filter_and_trim(waveform_slices_e, freqmin=freqmin, freqmax=freqmax)

    # rotating to r and t components
    event_baz = back_azimuth(event_lat, event_lon, sta_lat, sta_lon)
    
    waveform_slices_r_data, waveform_slices_t_data = [], []
    waveform_slices_r = Stream()
    waveform_slices_t = Stream()
    
    for i in range(len(waveform_slices_n)):
        rad, tran = rotate_ne_rt(waveform_slices_n[i].data, waveform_slices_e[i].data, event_baz)
        
        waveform_slice_r = waveform_slices_z[i].copy()
        waveform_slice_r.data = rad
        waveform_slice_r.stats.channel = 'LHR'
        waveform_slices_r += waveform_slice_r
        
        waveform_slice_t = waveform_slices_z[i].copy()
        waveform_slice_t.data = tran
        waveform_slice_t.stats.channel = 'LHT'
        waveform_slices_t += waveform_slice_t

        rad = np.asarray(rad)
        tran = np.asarray(tran)
        waveform_slices_r_data.append(rad)
        waveform_slices_t_data.append(tran)
        
    # downloading and filtering the event time
    event_begin_time = obspy.UTCDateTime(event_time) + event_time_to_arrive - 400
    event_end_time   = obspy.UTCDateTime(event_time) + event_time_to_arrive + 400
    event_waveform_z, event_waveform_n, event_waveform_e = acquire_waveform(network, station, inv, event_begin_time, event_end_time)

    event_waveform_z = filter_and_trim(event_waveform_z, freqmin=freqmin, freqmax=freqmax)
    event_waveform_n = filter_and_trim(event_waveform_n, freqmin=freqmin, freqmax=freqmax)
    event_waveform_e = filter_and_trim(event_waveform_e, freqmin=freqmin, freqmax=freqmax)

    event_waveform_r_data, event_waveform_t_data = rotate_ne_rt(event_waveform_n[0].data, event_waveform_e[0].data, event_baz)
    event_waveform_r = event_waveform_z.copy()  # Copy Z stream to get same timing and metadata
    event_waveform_r[0].data = event_waveform_r_data  # Replace data with R component
    event_waveform_r[0].stats.channel = 'LHR'  # Update channel name

    event_waveform_t = event_waveform_z.copy()  # Copy Z stream to get same timing and metadata
    event_waveform_t[0].data = event_waveform_t_data  # Replace data with T component
    event_waveform_t[0].stats.channel = 'LHT'  # Update channel name
    
    # finding the max amplitude of allowed event
    low_amplitude_slices_z, low_amplitude_data_z = noise_filterer(waveform_slices_z, event_waveform_z, expected_length=250, noise_max=noise_max_z)
    low_amplitude_slices_n, low_amplitude_data_n = noise_filterer(waveform_slices_n, event_waveform_n, expected_length=250, noise_max=noise_max_n)
    low_amplitude_slices_e, low_amplitude_data_e = noise_filterer(waveform_slices_e, event_waveform_e, expected_length=250, noise_max=noise_max_e)
    low_amplitude_slices_r, low_amplitude_data_r = noise_filterer(waveform_slices_r, event_waveform_r, expected_length=250, noise_max=noise_max_r)
    low_amplitude_slices_t, low_amplitude_data_t = noise_filterer(waveform_slices_t, event_waveform_t, expected_length=250, noise_max=noise_max_t)

    # plotting the noisy data
    plot_noisy_data(low_amplitude_slices_z, low_amplitude_slices_r, low_amplitude_slices_t, components='RT')

    return low_amplitude_data_z, low_amplitude_data_r, low_amplitude_data_t, low_amplitude_data_n, low_amplitude_data_e


def covariance_matrices(low_amplitude_data_z, low_amplitude_data_n, low_amplitude_data_e, components='RT'):
    ## showing the Covariance matrices
    Cd_z = np.cov(np.array(low_amplitude_data_z), rowvar=False)
    Cd_n = np.cov(np.array(low_amplitude_data_n), rowvar=False)
    Cd_e = np.cov(np.array(low_amplitude_data_e), rowvar=False)

    fig = plt.figure(figsize=(24, 6))
    axes = []

    ax_z = plt.subplot(1, 3, 1)
    Zplot = ax_z.imshow(Cd_z, cmap=plt.cm.cubehelix)
    plt.colorbar(Zplot, ax=ax_z)
    ax_z.contour(Cd_z, 10, colors='k')
    ax_z.set_title('Z component covariance matrix')

    ax_r = plt.subplot(1, 3, 2, sharey=ax_z)
    Rplot = ax_r.imshow(Cd_n, cmap=plt.cm.cubehelix)
    plt.colorbar(Rplot, ax=ax_r)
    ax_r.contour(Cd_e, 10, colors='k')

    ax_t = plt.subplot(1, 3, 3, sharey=ax_z)
    Tplot = ax_t.imshow(Cd_e, cmap=plt.cm.cubehelix)
    plt.colorbar(Tplot, ax=ax_t)
    ax_t.contour(Cd_e, 10, colors='k')

    if components == 'NE':
        ax_r.set_title('N component covariance matrix')
        ax_t.set_title('E component covariance matrix')

    elif components == 'RT':
        ax_r.set_title('R component covariance matrix')
        ax_t.set_title('T component covariance matrix')

    axes.append(ax_z)
    axes.append(ax_r)
    axes.append(ax_t)

    plt.show()
    return Cd_z, Cd_n, Cd_e

def inverse_covariance_matrices(Cd_z, Cd_n, Cd_e):
    epsilon = 1e-12

    C_ddz = Cd_z + np.eye(Cd_z.shape[0]) * epsilon
    C_ddn = Cd_n + np.eye(Cd_n.shape[0]) * epsilon
    C_dde = Cd_e + np.eye(Cd_e.shape[0]) * epsilon

    Cddinv_z = np.linalg.inv(C_ddz)
    Cddinv_n = np.linalg.inv(C_ddn)
    Cddinv_e = np.linalg.inv(C_dde)

    return Cddinv_z, Cddinv_n, Cddinv_e, C_ddz, C_ddn, C_dde

def log_inverse_determinant(C_ddz, C_ddn, C_dde):
    C_ddz_sum, C_ddn_sum, C_dde_sum = 0, 0, 0
    for i in range(len(np.linalg.eig(C_ddz)[0])):
        C_ddz_sum += np.log(np.linalg.eig(C_ddz)[0][i])
    for i in range(len(np.linalg.eig(C_ddn)[0])):
        C_ddn_sum += np.log(np.linalg.eig(C_ddn)[0][i])
    for i in range(len(np.linalg.eig(C_dde)[0])):
        C_dde_sum += np.log(np.linalg.eig(C_dde)[0][i])

    print("C_ddz_sum:", C_ddz_sum)
    print("C_ddn_sum:", C_ddn_sum)
    print("C_dde_sum:", C_dde_sum)

    Cddinv_z_logdet = 0.5*(-C_ddz_sum).real
    Cddinv_n_logdet = 0.5*(-C_ddn_sum).real
    Cddinv_e_logdet = 0.5*(-C_dde_sum).real

    print("c_ddz_inv_det:", Cddinv_z_logdet)
    print("c_ddn_inv_det:", Cddinv_n_logdet)
    print("c_dde_inv_det:", Cddinv_e_logdet)

    return Cddinv_z_logdet, Cddinv_n_logdet, Cddinv_e_logdet


def component_finder(low_amplitude_data_z, low_amplitude_data_n, low_amplitude_data_e, components='RT'):
    '''
    n and e can be cleanly replaced with r and t
    returns Cdinv_z, Cdinv_n, Cdinv_e, logdetCdinv_z, logdetCdinv_n, logdetCdinv_e
    '''
    Cdz, Cdn, Cde = covariance_matrices(low_amplitude_data_z, low_amplitude_data_n, low_amplitude_data_e, components=components)
    
    Cddinv_z, Cddinv_n, Cddinv_e, C_ddz, C_ddn, C_dde = inverse_covariance_matrices(Cdz, Cdn, Cde)
    
    Cddinv_z_logdet, Cddinv_n_logdet, Cddinv_e_logdet = log_inverse_determinant(C_ddz, C_ddn, C_dde)
    
    return Cddinv_z, Cddinv_n, Cddinv_e, Cddinv_z_logdet, Cddinv_n_logdet, Cddinv_e_logdet