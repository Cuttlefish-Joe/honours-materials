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

## basic functions
def distance(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0]
def back_azimuth(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[2]




## waveform stuff
def acquire_waveform_son(network, station, inv, start_time, end_time):
    waveform_z = ap.get_waveforms(network=network, station=station, location='*', channel='LHZ', starttime=start_time, endtime=end_time)
    waveform_z.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.15)
    waveform_z.detrend('linear')
    waveform_z.detrend('demean')
    waveform_z.taper(max_percentage=0.1)

    waveform_n = ap.get_waveforms(network=network, station=station, location='*', channel='LHN', starttime=start_time, endtime=end_time)
    waveform_n.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.15)
    waveform_n.detrend('linear')
    waveform_n.detrend('demean')
    waveform_n.taper(max_percentage=0.1)

    waveform_e = ap.get_waveforms(network=network, station=station, location='*', channel='LHE', starttime=start_time, endtime=end_time)
    waveform_e.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.15)
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

def filter_and_trim_son(waveform_slices, freqmin=0.04, freqmax=0.06):
    for i in range(len(waveform_slices)):
        waveform_slices[i].filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=3, zerophase=True)
    for i in range(len(waveform_slices)):
        waveform_slices[i].trim(starttime=waveform_slices[i].stats.starttime + 200, endtime=waveform_slices[i].stats.endtime - 200)
        waveform_slices[i].taper(max_percentage=0.15)
    return waveform_slices

def noise_filterer(waveform_slices, event_waveform, expected_length=400, noise_max=15):
    low_amplitude_slices = Stream()
    low_amplitude_data = []

    max_amplitude = max(abs(event_waveform[0].data)) # will probably have to change this for multiple stations
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

six_hours = 60*60*6
half_hour = 60*30

def rsnss(station_info, event_info, slice_duration=800, overlap=400, freqmin=0.04, freqmax=0.06, sampling_length=60*60*6, sampling_start=60*30, noise_max_z=15, noise_max_r=15, noise_max_t=15, noise_max_n=15, noise_max_e=15, components= 'RT'):
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
    
    waveform_z, waveform_n, waveform_e = acquire_waveform_son(network, station, inv, begin_time, end_time)

    #waveform_z.plot(type='dayplot')
    
    waveform_trim_z = filter_and_trim_son(waveform_z, freqmin=freqmin, freqmax=freqmax)
    waveform_trim_n = filter_and_trim_son(waveform_n, freqmin=freqmin, freqmax=freqmax)
    waveform_trim_e = filter_and_trim_son(waveform_e, freqmin=freqmin, freqmax=freqmax)
    
    waveform_slices_z = split_waveform_overlapping(waveform_trim_z, slice_duration=slice_duration, overlap=overlap)
    waveform_slices_n = split_waveform_overlapping(waveform_trim_n, slice_duration=slice_duration, overlap=overlap)
    waveform_slices_e = split_waveform_overlapping(waveform_trim_e, slice_duration=slice_duration, overlap=overlap)
    
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
    event_waveform_z, event_waveform_n, event_waveform_e = acquire_waveform_son(network, station, inv, event_begin_time, event_end_time)

    event_waveform_z = filter_and_trim_son(event_waveform_z, freqmin=freqmin, freqmax=freqmax)
    event_waveform_n = filter_and_trim_son(event_waveform_n, freqmin=freqmin, freqmax=freqmax)
    event_waveform_e = filter_and_trim_son(event_waveform_e, freqmin=freqmin, freqmax=freqmax)

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
    if components=='RT':
        plot_noisy_data(low_amplitude_slices_z, low_amplitude_slices_r, low_amplitude_slices_t, components=components)
        
    elif components=='NE':
        plot_noisy_data(low_amplitude_slices_z, low_amplitude_slices_n, low_amplitude_slices_e, components=components)

    return low_amplitude_data_z, low_amplitude_data_r, low_amplitude_data_t, low_amplitude_data_n, low_amplitude_data_e

def random_noise_samples(low_amps_z, low_amps_r, low_amps_t, n_samples=20, components='RT'):
    fig = plt.figure(figsize=(24, 6))
    axes = []

    ax_z = plt.subplot(1, 3, 1)
    ax_n = plt.subplot(1, 3, 2)
    ax_e = plt.subplot(1, 3, 3)

    total_realisations_z = len(low_amps_z[0])
    total_realisations_r = len(low_amps_r[0])
    total_realisations_t = len(low_amps_t[0])

    random_indices_z = np.random.choice(total_realisations_z, n_samples, replace=False)
    random_indices_r = np.random.choice(total_realisations_r, n_samples, replace=False)
    random_indices_t = np.random.choice(total_realisations_t, n_samples, replace=False)

    time_axis = np.arange(len(low_amps_z[0][0]))

    for i, idx in enumerate(random_indices_z):
        ax_z.plot(time_axis, low_amps_z[0][idx], alpha=0.7, label=f'Realization {idx}')

    for i, idx in enumerate(random_indices_r):
        ax_n.plot(time_axis, low_amps_r[0][idx], alpha=0.7, label=f'Realization {idx}')

    for i, idx in enumerate(random_indices_t):
        ax_e.plot(time_axis, low_amps_t[0][idx], alpha=0.7, label=f'Realization {idx}')

    ax_z.set_xlabel('Sample')
    ax_z.set_ylabel('Amplitude')
    ax_n.set_xlabel('Sample')
    ax_n.set_ylabel('Amplitude')
    ax_e.set_xlabel('Sample')
    ax_e.set_ylabel('Amplitude')
    ax_z.grid(True, alpha=0.3)
    ax_n.grid(True, alpha=0.3)
    ax_e.grid(True, alpha=0.3)

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



## covariance shit
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

def create_diagonal_matrix_from_average(matrix: np.ndarray, keep_elements: int = None) -> np.ndarray:
    """
    Calculates a new matrix where each diagonal is the average of the
    corresponding diagonal in the input matrix.
    This function iterates through all diagonals (main, upper, and lower) of the
    input matrix, calculates the average of the elements on each diagonal, and
    then constructs a new matrix using these average values to populate the
    corresponding diagonals.
    An optional `keep_elements` parameter can be provided to specify the number
    of non-zero diagonals to keep, centered around the main diagonal.
    Args:
        matrix: A square NumPy array (matrix).
        keep_elements: An integer specifying the number of non-zero diagonals
                       to keep. A value of 1 keeps only the main diagonal.
                       If None, all diagonals are kept.
    Returns:
        A new NumPy array with each diagonal replaced by the average of the
        original's corresponding diagonal.
    Raises:
        ValueError: If the input matrix is not square or if keep_elements is
                    invalid.
    """
    # Check if the input is a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    n = matrix.shape[0]
    # Create an empty matrix to build the result
    new_matrix = np.zeros_like(matrix, dtype=np.float64)
    # Determine the range of diagonals to process
    if keep_elements is None:
        diagonals_to_keep_each_side = n - 1
    else:
        if not isinstance(keep_elements, int) or keep_elements < 1 or keep_elements > 2 * n - 1:
            raise ValueError("keep_elements must be a positive integer "
                             "less than or equal to the total number of diagonals.")
        diagonals_to_keep_each_side = (keep_elements - 1) // 2
    # Iterate through only the specified diagonals and populate the new matrix
    for k in range(-diagonals_to_keep_each_side, diagonals_to_keep_each_side + 1):
        diagonal_elements = np.diag(matrix, k=k)
        if diagonal_elements.size > 0:
            average_value = np.mean(diagonal_elements)
            # Create a temporary matrix with the average on the current diagonal
            temp_diag = np.diag(np.full_like(diagonal_elements, average_value), k=k)
            new_matrix += temp_diag
    return new_matrix

def meaning_matrices(Cd_list):
    max_list = []
    for i in range(len(Cd_list)):
        max_list.append(np.max(np.abs(Cd_list[i])))
        
    mean_max = np.mean(max_list)
    print('mean max', mean_max)
    
    for i in range(len(Cd_list)):
        Cd_list[i] = Cd_list[i] * (mean_max / max_list[i])
        print('new max', np.max(np.abs(Cd_list[i])))
        
    return Cd_list


def plot_2_Cds(Cd_z, Cd_r, Cd_t, Cd_z2, Cd_r2, Cd_t2):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    test_checker_x = np.linspace(-150, 100, 250)


    # Create figure and gridspec with height ratios
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.1, 
                        height_ratios=[2, 2, 1])  # Top two rows twice as tall as bottom row

    # Top row - 3 subplots (tall)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    Zplot = ax1.imshow(Cd_z, cmap=plt.cm.managua)
    plt.colorbar(Zplot, ax=ax1)
    ax1.contour(Cd_z, 5, colors='k')
    ax1.set_title('Z covariance matrix (original)', c='peru')

    Rplot = ax2.imshow(Cd_r, cmap=plt.cm.managua) # plt.cm.cubehelix
    plt.colorbar(Rplot, ax=ax2)
    ax2.contour(Cd_r, 5, colors='k')
    ax2.set_title('R covariance matrix (original)', c='peru')

    Tplot = ax3.imshow(Cd_t, cmap=plt.cm.managua)
    plt.colorbar(Tplot, ax=ax3)
    ax3.contour(Cd_t, 5, colors='k')
    ax3.set_title('T covariance matrix (original)', c='peru')

    # Middle row - 3 subplots (tall)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    Zplot_filt = ax4.imshow(Cd_z2, cmap=plt.cm.managua)
    plt.colorbar(Zplot_filt, ax=ax4)
    ax4.contour(Cd_z2, 5, colors='k')
    ax4.set_title('Z covariance matrix (scaled)', c='cornflowerblue')

    Rplot_filt = ax5.imshow(Cd_r2, cmap=plt.cm.managua)
    plt.colorbar(Rplot_filt, ax=ax5)
    ax5.contour(Cd_r2, 5, colors='k')
    ax5.set_title('R covariance matrix (scaled)', c='cornflowerblue')

    Tplot_filt = ax6.imshow(Cd_t2, cmap=plt.cm.managua)
    plt.colorbar(Tplot_filt, ax=ax6)
    ax6.contour(Cd_t2, 5, colors='k')
    ax6.set_title('T covariance matrix (scaled)', c='cornflowerblue')

    # Bottom row - 3 subplots (short)
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    axes = [ax7, ax8, ax9]

    n_stations = 1 ## << we want these two realisations to look similar, 

    for i in range(n_stations):
        for j, comp in enumerate(['Z', 'R', 'T']):
            ax = axes[j]
            ax.plot(test_checker_x, np.random.multivariate_normal(np.zeros(250), Cd_z if comp == 'Z' else Cd_r if comp == 'R' else Cd_t), c='peru', label='Original Cov')
            # ax.plot(test_checker_x, np.random.multivariate_normal(np.zeros(250), smoothed_Cd_z_symmetric if comp == 'Z' else smoothed_Cd_r_symmetric if comp == 'R' else smoothed_Cd_t_symmetric), c='c', label='Smoothed Cov')
            ax.plot(test_checker_x, np.random.multivariate_normal(np.zeros(250), Cd_z2 if comp == 'Z' else Cd_r2 if comp == 'R' else Cd_t2), c='cornflowerblue', label='Smoothed Symmetric Cov')
            ax.set_ylabel(f'{comp}')
            if i == n_stations - 1:
                ax.set_xlabel('Time (s)')
            if i == 0:
                ax.set_title(comp)
            ax.grid(True)

    # Add sample plots to each subplot
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    plt.suptitle('Data-Based Covariance Matrices and Random Noise Realizations \n brown: unedited, blue: processed', fontsize=16)
    plt.tight_layout()
    plt.show()
    return fig, axes

def inverse_and_det(Cd_z, Cd_r, Cd_t):
    Cd_z_inv = np.linalg.inv(Cd_z)
    Cd_r_inv = np.linalg.inv(Cd_r)
    Cd_t_inv = np.linalg.inv(Cd_t)

    Cd_z_sum, Cd_r_sum, Cd_t_sum = 0, 0, 0

    for i in range(len(np.linalg.eig(Cd_z)[0])):
        Cd_z_sum += np.log(np.linalg.eig(Cd_z)[0][i])
    for i in range(len(np.linalg.eig(Cd_r)[0])):
        Cd_r_sum += np.log(np.linalg.eig(Cd_r)[0][i])
    for i in range(len(np.linalg.eig(Cd_t)[0])):
        Cd_t_sum += np.log(np.linalg.eig(Cd_t)[0][i])

    print('Cd_z_sum: ', Cd_z_sum)
    print('Cd_r_sum: ', Cd_r_sum)
    print('Cd_t_sum: ', Cd_t_sum)

    Cd_z_inv_logdet = 0.5*(-Cd_z_sum.real)
    Cd_r_inv_logdet = 0.5*(-Cd_r_sum.real)
    Cd_t_inv_logdet = 0.5*(-Cd_t_sum.real)

    print('Cd_z_inv_logdet: ', Cd_z_inv_logdet) 
    print('Cd_r_inv_logdet: ', Cd_r_inv_logdet)
    print('Cd_t_inv_logdet: ', Cd_t_inv_logdet)
    
    return Cd_z_inv, Cd_r_inv, Cd_t_inv, Cd_z_inv_logdet, Cd_r_inv_logdet, Cd_t_inv_logdet

## data downloading
def acquire_real_waveforms(station_info, event_info, inv):
    event_lat, event_lon, event_time = event_info
    sta_lats, sta_lons, networks_flat, stations_flat = station_info

    real_signals = Stream()
    real_signals.clear()

    dist_infos, back_azs, begin_times, end_times, arrival_times = [], [], [], [], []

    for i in range(len(sta_lats)):
        dist_info = distance(event_lat, event_lon, sta_lats[i], sta_lons[i]) / 1000
        back_az = back_azimuth(event_lat, event_lon, sta_lats[i], sta_lons[i])
        
        try:
            time_to_arrive = dist_info / 2.95
            begin_time = obspy.UTCDateTime(event_time) + time_to_arrive - 400  # 350s before (400s)
            end_time = obspy.UTCDateTime(event_time) + time_to_arrive + 400  # 300s after
            
            arrival_time = obspy.UTCDateTime(event_time) + time_to_arrive
            
            st = ap.get_waveforms(network=networks_flat[i], station=stations_flat[i], location='*', channel="LHZ,LHN,LHE", starttime=begin_time, endtime=end_time)
            st.remove_response(inventory=inv, output="DISP", taper=True, taper_fraction=0.15)
            for tr in st:
                tr.stats.update({"arrival": arrival_time})
                tr.stats.update({"baz": back_az})
                tr.stats.update({"distancey": dist_info})
                tr.stats.update({"idx": i})

            real_signals += st

            dist_infos.append(dist_info)
            back_azs.append(back_az)
            begin_times.append(begin_time)
            end_times.append(end_time)
            arrival_times.append(arrival_time)

        except Exception as e:
            print(f"Error processing station {i}: {e}")

    return real_signals, dist_infos, back_azs, begin_times, end_times, arrival_times

def preprocessing_real_waveforms(real_signals, window=250, freqmin=0.04, freqmax=0.06):
    # pre-process / prepare the data
    window = 250

    npts = np.min([_.stats.npts for _ in real_signals])
    delta = real_signals[0].stats.delta

    real_signals.detrend('linear')
    real_signals.detrend('demean')
    real_signals.taper(max_percentage=0.1)
    real_signals.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax, corners=3, zerophase=True)

    # isolating the surface window around arrivals
    for tr in real_signals.select(component='Z'):
        st = real_signals.select(network=tr.stats.network, station=tr.stats.station)
        st.trim(tr.stats.arrival-window*0.6, tr.stats.arrival+window*0.4)
        st.rotate('NE->RT', back_azimuth=tr.stats.baz)
    for tr in real_signals:
        tr.stats.update({"offset": tr.stats.starttime-tr.stats.arrival})

    real_signals.taper(max_percentage=0.15)
    
    return real_signals, npts, delta
    
def setup_ydata(real_signals, stationse=None):
    real_station_streams = []

    if(stationse):
        stations = stationse
    else:
        stations = list(set([tr.stats.station for tr in real_signals]))

    for station in stations:
        # Get traces for this station
        station_stream = real_signals.select(station=station)
        
        if len(station_stream) == 3:
            # Sort by component to ensure Z, R, T order
            station_stream = station_stream.select(component='Z') + \
                            station_stream.select(component='R') + \
                            station_stream.select(component='T')
            
        real_station_streams.append(station_stream)   
        
        
    # creating the ydatas data format
    data_len = len(real_station_streams[0][0])
    ydatas = []

    for i in range(len(real_station_streams)):
        ydata = np.zeros((data_len, 3))
        ydata[:, 0] = real_station_streams[i][0].data
        ydata[:, 1] = real_station_streams[i][1].data
        ydata[:, 2] = real_station_streams[i][2].data
        ydatas.append(ydata)
        
    return ydatas, real_station_streams


## elementary things
def stf_tsai2007(source_duration, deltat):
    halfT = source_duration/2
    N = int(halfT/deltat)
    tvec = np.arange(-2*N, 2*N+1) * deltat
    STF = np.zeros_like(tvec)
    STF[(-halfT<tvec) * (tvec<0)] = 1
    STF[(0<tvec) * (tvec<halfT)] = -1
    return STF

def convolution_function(datastream, timedur=15, model='prem_a_10s'):
    coche = client.get_model_info(model_name=model)
    
    new_sliprate = stf_tsai2007(source_duration=timedur, deltat=coche['dt'])
    stf_decov_f = np.fft.rfft(coche.sliprate, coche.nfft)
    mask = (np.abs(stf_decov_f) > 1e-6*np.max(np.abs(stf_decov_f)))
    stf_cov_f = np.fft.rfft(new_sliprate, coche.nfft)
    stf_recov_f = stf_cov_f / stf_decov_f
    stf_recov_f[~mask] = 0 + 0j    
    
    for tr in datastream:
        dota = np.fft.irfft(stf_recov_f*np.fft.rfft(tr.data, coche.nfft))
        tr.data = dota[:coche.npts]
        tr.stats.starttime += timedur/2
        
    return datastream

def station_residual(d_obs, d_pred, max_shift=0, shifts_per_station=3, return_tshift=False):
    '''
    Compute the residual between observed and predicted data for a 3-component station.
    Parameters
    ----------
    d_obs : ndarray
        Observed data, shape (3, npts), shorted by Z, R, T components
    d_pred : ndarray
        Predicted data, shape (3, npts)
    max_shift : int, optional
        Maximum time shift (in samples) to consider for cross-correlation. Default is 0.
    shifts_per_station : int, optional
        Number of time shifts to compute per station. Can be 1, 2, or 3. Default is 3.
    return_tshift : bool, optional
        If True, return the list of time shifts applied. Default is False.
    Returns
    -------
    residual : ndarray
        Residual data after applying time shifts, shape (3, npts).
    tshift : list
        List of time shifts applied to each component, length 3.
    '''
    assert d_obs.shape == d_pred.shape
    assert d_obs.shape[0] == 3
    npts = d_obs.shape[1]
    if max_shift >= npts:
        raise ValueError('max_shift must be less than the number of points in the data.')
    if max_shift <= 0: # return pointwise difference if no shift is allowed
        return d_obs - d_pred, [0]*3 if return_tshift else d_obs - d_pred
    ## Compute the cross-correlation for time shifts
    f_pred = np.fft.rfft(d_pred, 2*npts-1)
    f_obs = np.fft.rfft(d_obs, 2*npts-1)
    d_xcorr = np.fft.fftshift(np.fft.irfft(f_pred * np.conj(f_obs), 2*npts-1), axes=1)
    d_xcorr = d_xcorr[:, npts-1-max_shift:npts+max_shift] # only keep relevant shifts
    t_xcorr = np.arange(-max_shift, max_shift+1)
    # t_xcorr = np.arange(-npts+1, npts)
    # print ('t_xcorr', t_xcorr)
    # print ('t_xcorr', np.arange(-npts+1, npts)[npts-1-max_shift:npts+max_shift])
    ## Determine time shifts based on shifts_per_station
    if shifts_per_station == 3: # time shift for each component independently
        tshift = [t_xcorr[np.argmax(_)] for _ in d_xcorr]
        # print ('tshift', tshift)
    elif shifts_per_station == 2: # same shift for Z and R, different for T
        tshift = [t_xcorr[np.argmax(d_xcorr[0] + d_xcorr[1])]] * 2 # same shift for Z and R
        tshift.append(t_xcorr[np.argmax(d_xcorr[2])]) # independent shift for T
    elif shifts_per_station == 1: # same shift for all components
        tshift = [t_xcorr[np.argmax(d_xcorr[0] + d_xcorr[1] + d_xcorr[2])]] * 3 # same shift for all
    else:
        return ValueError('Invalide shifts_per_station!')

    ## Apply time shifts to predicted data
    residual = np.zeros_like(d_obs)
    for i in range(3):
        # print ('obs', d_obs[i])
        # print ('pred', np.roll(d_pred[i], tshift[i]))
        residual[i] = d_obs[i] - np.roll(d_pred[i], -tshift[i])
        # print ('res', residual[i])
    if return_tshift:
        return residual, tshift
    else:
        return residual


def SF_elementary_func(set, bulk_stations, event_info, time_info, scaling_SF):
    event_lat, event_lon, event_time = event_info
    starter_time, ttot, arrival_times_expanded = time_info
    
    data = client.get_waveforms_bulk(
        model = 'prem_a_10s',
        bulk = bulk_stations,
        sourcelatitude = event_lat,
        sourcelongitude = event_lon,
        sourcedepthinmeters = 500,  # 500 m depth
        sourceforce = set*scaling_SF, # single force event - Fr, Ft, Fp
        origintime = obspy.UTCDateTime(event_time),
        starttime = -starter_time,
        endtime = ttot, 
        components = 'ZRT',
        units = 'displacement'
    )
    
    conv_data = convolution_function(data)
    
    conv_data.resample(1.0)

    for tr in conv_data:
        # correct for sign flip in the radial component
        if tr.stats.channel.endswith('R'): tr.data *= -1

    conv_data.detrend('demean')
    conv_data.detrend('linear')
    conv_data.taper(max_percentage=0.1)
    
    for i in range(len(conv_data)):
        conv_data[i].filter('bandpass', freqmin = 0.04, freqmax = 0.06, corners = 3, zerophase = True) # ok looks like this happens post conv.
        conv_data[i].trim(starttime=arrival_times_expanded[i]-150, endtime=arrival_times_expanded[i]+100)
        conv_data[i].taper(max_percentage=0.15)
        #data[i].trim(starttime=data[i].stats.starttime + 200, endtime=data[i].stats.endtime - 200)
        
        # Ensure exactly 250 samples
        if len(conv_data[i].data) > 250:
            # Trim to first 250 samples
            conv_data[i].data = conv_data[i].data[:250]
            conv_data[i].stats.npts = 250
        elif len(conv_data[i].data) < 250:
            # Pad with zeros to reach 250 samples
            padding = np.zeros(250 - len(conv_data[i].data))
            conv_data[i].data = np.concatenate([conv_data[i].data, padding])
            conv_data[i].stats.npts = 250

        # # Update endtime to match new length
        # data[i].stats.endtime = data[i].stats.starttime + (399 / data[i].stats.sampling_rate)
    
    station_streems = []
    outputs = []
    # Fix: Extract station codes from trace stats, not the traces themselves
    
    stations = list(set([tr.stats.station for tr in conv_data]))
    
    for station in stations:
        station_streem = conv_data.select(station=station)
        if len(station_streem) == 3:
            station_streem = station_streem.select(component='Z') + \
                        station_streem.select(component='R') + \
                        station_streem.select(component='T')
            station_streems.append(station_streem)

            ydota = [station_streem[0].data, station_streem[1].data, station_streem[2].data]            
            outputs.append(ydota)

    return outputs


def MT_elementary_func(set, bulk_stations, event_info, time_info, scaling_MT):
    event_lat, event_lon, event_time = event_info
    starter_time, ttot, arrival_times_expanded = time_info
    
    data = client.get_waveforms_bulk(
        model = 'prem_a_10s',
        bulk = bulk_stations,
        sourcelatitude = event_lat,
        sourcelongitude = event_lon,
        sourcedepthinmeters = 500,  # 500 m depth
        sourcemomenttensor = set*scaling_MT, # single force event - Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
        origintime = obspy.UTCDateTime(event_time),
        starttime = -starter_time,
        endtime = ttot, 
        components = 'ZRT',
        units = 'displacement'
    )
    data.resample(1.0)
    data.detrend('demean')
    data.detrend('linear')
    data.taper(max_percentage=0.1)
    
    for i in range(len(data)): # no convolution required for MT sources apparently
        data[i].trim(starttime=arrival_times_expanded[i]-350, endtime=arrival_times_expanded[i]+300)
        data[i].filter('bandpass', freqmin = 0.04, freqmax = 0.06, corners = 3, zerophase = True)
        data[i].trim(starttime=data[i].stats.starttime + 200, endtime=data[i].stats.endtime - 200)
        data[i].taper(max_percentage=0.15)
        
        # Ensure exactly 250 samples
        if len(data[i].data) > 250:
            # Trim to first 250 samples
            data[i].data = data[i].data[:250]
            data[i].stats.npts = 250
        elif len(data[i].data) < 250:
            # Pad with zeros to reach 250 samples
            padding = np.zeros(250 - len(data[i].data))
            data[i].data = np.concatenate([data[i].data, padding])
            data[i].stats.npts = 250

        # # Update endtime to match new length
        # data[i].stats.endtime = data[i].stats.starttime + (399 / data[i].stats.sampling_rate)
    
    station_streems = []
    outputs = []
    # Fix: Extract station codes from trace stats, not the traces themselves
    
    stations = list(set([tr.stats.station for tr in data]))

    for station in stations:
        station_streem = data.select(station=station)
        if len(station_streem) == 3:
            station_streem = station_streem.select(component='Z') + \
                        station_streem.select(component='R') + \
                        station_streem.select(component='T')
            station_streems.append(station_streem)

            ydota = [station_streem[0].data, station_streem[1].data, station_streem[2].data]            
            outputs.append(ydota)

    return outputs

## trans-C

def log_prior(x, state, mu, cov, cov_dets):
    '''
    Normalised normal distribution prior for the parameters of the source models. 
    Assumes no 1D states and that the covariance matrix is not inverse. 
    This one is not impacted by multiple stations being at play. 
    '''
    mean = mu[state]
    covar = cov[state]
    covar_det = cov_dets[state]
    r = mean - x
    
    log_const = -0.5*len(x)*np.log(2*np.pi) - 0.5*np.log(covar_det)
    log_exp = -0.5 * np.dot(r, np.linalg.solve(covar, r))
    
    return log_const + log_exp

def log_likelihood(x, state, y, cdinv, log_cdinv_dets, Glist, align=True, verbose=False):
    '''
    NEEDS STATIONS DEFINED BEFOREHAND
    Gaussian log likelihood function - requires you to input the "G matrix" for each state you want to test
    y is the data going in. This is a sum of the log likelihoods for each component (Z, R, T).
    This one IS impacted by multiple stations being at play, and the log likelihoods from each are summed here.  
    '''
    ll_z_sum, ll_r_sum, ll_t_sum = 0, 0, 0
    
    for i in range(len(stations)):
    
        G = Glist[i][state]
        Cdinv_z, Cdinv_r, Cdinv_t = cdinv[i][state]
        Cdinv_det_z, Cdinv_det_r, Cdinv_det_t = log_cdinv_dets[i][state]
        data_z, data_r, data_t = y[i][:, 0], y[i][:, 1], y[i][:, 2]
        model_z, model_r, model_t = np.dot(G[:,0], x), np.dot(G[:,1], x), np.dot(G[:,2], x)
        
        if(align):
            d_obs = np.vstack((data_z, data_r, data_t))
            d_pred = np.vstack((model_z, model_r, model_t))
            res, tshifts = station_residual(d_obs, d_pred, max_shift=30, shifts_per_station=3, return_tshift=True)
            if(verbose):
                print(' Station ', i, ' offsets detected ', tshifts)
            aligned = np.zeros_like(d_pred)
            for j in range(3): aligned[j] = np.roll(d_pred[j], tshifts[j])
            model_z, model_r, model_t = aligned
        
        r_z = data_z - model_z
        r_r = data_r - model_r
        r_t = data_t - model_t

        ll_z = -.5 * r_z @ Cdinv_z @ r_z.T
        ll_r = -.5 * r_r @ Cdinv_r @ r_r.T 
        ll_t = -.5 * r_t @ Cdinv_t @ r_t.T

        # ll_z += -0.5*len(data_z)*np.log(2*np.pi) - 0.5*Cdinv_det_z
        # ll_r += -0.5*len(data_r)*np.log(2*np.pi) - 0.5*Cdinv_det_r
        # ll_t += -0.5*len(data_t)*np.log(2*np.pi) - 0.5*Cdinv_det_t

        ll_z_sum += ll_z
        ll_r_sum += ll_r
        ll_t_sum += ll_t

    return ll_z_sum + ll_r_sum + ll_t_sum

# setting up the log posterior function
def log_posterior(x, state, y, cdinv, log_cdinv_dets, Glist, mu, cov, cov_dets):
    '''
    log posterior = log likelihood + log prior
    '''
    lp = log_likelihood(x, state, y, cdinv, log_cdinv_dets, Glist) 
    lp += log_prior(x,state,mu,cov, cov_dets)
    return lp


def ml_optimiser(log_posterior_args, nstates):
    ydatas, cdinv, log_cdinv_dets, Glist, mu, cov, cov_dets = log_posterior_args    

    ml = []
    print("Maximum likelihood estimates:")
    for i in range(nstates):
        obj_fun = lambda *args: -log_posterior(*args)
        initial = np.array(mu[i]) # arbitrary initial guesses for optimisation
        # you can play about with this number to see if it matters much
        soln = minimize(obj_fun, initial, args=(i, ydatas, cdinv, log_cdinv_dets, Glist, mu, cov, cov_dets), method='nelder-mead')
        #print(soln)
        ml.append(soln.x)
        print(soln.x)
        #print("x_ml = {0:.3f}".format(soln.x[0]))
    return ml


def print_diagnostics(tcs3, elapsed_time):
    # print some diagnostics
    print('\n Algorithm type                                      :', tcs3.alg)
    print(' Average % acceptance rate for within states         :',np.round(tcs3.accept_within,2))
    print(' Average % acceptance rate for between states        :',np.round(tcs3.accept_between,2))

    # extract trans-D samples and chains
    discard = 0                  # chain burnin
    thin = 15                    # chain thinning
    chain,states_chain = tcs3.get_visits_to_states(discard=discard,thin=thin,normalize=True,
                                                walker_average='median',return_samples=True)

    print(' Auto correlation time for between state sampling    :',np.round(tcs3.autocorr_time_for_between_state_jumps,3))
    print(' Total number of state changes for all walkers       :',tcs3.total_state_changes)
    #print(' Number of state changes for each walker             :\n',*tcs3.state_changes_perwalker)
    #print(' True relative marginal Likelihoods                  :', *trueML)
    print(' Estimated relative evidences                        :', *np.round((tcs3.relative_marginal_likelihoods),5))
    print(' Elapsed time                                        :', np.round(elapsed_time,2),'s \n')
    return chain, states_chain

def corner_plots(ensemble_per_state, mu, nstates, ndims):
    contour_kwargs = {"linewidths" : 0.5}
    data_kwargs = {"color" : "darkblue"}
    data_kwargs = {"color" : "slateblue"}
    for i in range(nstates):
        string = 'State '+str(i)
        print(' State; ',i,' in ',ndims[i],' dimensions')
        fig = corner.corner(
            ensemble_per_state[i], 
            truths=mu[i], # << my Mu aren't truths but maybe this can help calibrate them a bit
            title=string,
            bins=40,hist_bin_factor=2,smooth=True,contour_kwargs=contour_kwargs,data_kwargs=data_kwargs
            );
    return fig


def plot_ratios(tcs3, chain, states_chain, nstates, nsamples=200000):
    key = tcs3
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
    if(chain.ndim == 3): # plot chains for each state and walker
        ax1.semilogx(chain.reshape(np.shape(chain)[0],-1),lw=0.75)
    elif(chain.ndim==2): # plot chains for each state average over walkers
        ax1.semilogx(chain.reshape(np.shape(chain)[0],-1),lw=0.75,label=['State 1','State 2','State 3'])
        ax1.legend()
    ax1.set_xlabel('Chain step')
    ax1.set_ylabel('Relative Evidence')
    ax1.set_title(' Convergence of algorithm: '+key.alg)
    ax1.set_ylim(0.0,0.8)
    transc_ensemble,model_chains,states_chain = key.get_transc_samples(nsamples,returnchains=True,verbose=False)
    h = np.zeros(key.nstates)
    h[list(Counter(states_chain.reshape(-1)).keys())] = list(Counter(states_chain.reshape(-1)).values())
    h /= np.sum(h)
    labels = ['State '+str(i+1) for i in np.arange(nstates)]
    x = np.arange(nstates)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    offset = width * multiplier
    rects = ax2.bar(x + offset, np.round(h,3), width, label=key.alg,color='skyblue')
    ax2.bar_label(rects, padding=3)
    multiplier += 1
    offset = width * multiplier+0.05
    ax2.set_ylabel(' Proportion of visits to each state')
    ax2.set_title('Relative Evidence')
    ax2.set_xticks(x + width/2, labels)
    # ax2.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    return fig, ax1, ax2

def MAP_stuff(tcs3, scaling_SF, scaling_MT, SFs, MTs, other_info):
    SF_e1, SF_e2, SF_e3 = SFs
    MT_e1, MT_e2, MT_e3, MT_e4, MT_e5, MT_e6 = MTs
    mu, cov, cov_dets, ydatas, cdinv, log_cdinv_dets, Glist, stations = other_info
    
    max_arg_SF = np.argmax(tcs3.log_posterior_ens[0])
    print(max_arg_SF)
    max_arg_MT = np.argmax(tcs3.log_posterior_ens[1])
    print(max_arg_MT)
    min_arg_SF = np.argmin(tcs3.log_posterior_ens[0])
    print(min_arg_SF)
    min_arg_MT = np.argmin(tcs3.log_posterior_ens[1])
    print(min_arg_MT)
    
    print(f'log posterior SF max: {tcs3.log_posterior_ens[0][max_arg_SF]}')
    print(f'log prior + log likelihood SF max: {log_prior(tcs3.ensemble_per_state[0][max_arg_SF], 0, mu, cov, cov_dets) + log_likelihood(tcs3.ensemble_per_state[0][max_arg_SF], 0, ydatas, cdinv, log_cdinv_dets, Glist)}')
    print(f'log posterior MT max: {tcs3.log_posterior_ens[1][max_arg_MT]}')
    print(f'log prior + log likelihood MT max: {log_prior(tcs3.ensemble_per_state[1][max_arg_MT], 1, mu, cov, cov_dets) + log_likelihood(tcs3.ensemble_per_state[1][max_arg_MT], 1, ydatas, cdinv, log_cdinv_dets, Glist)}')
    print(f'log posterior SF min: {tcs3.log_posterior_ens[0][min_arg_SF]}')
    print(f'log prior + log likelihood SF min: {log_prior(tcs3.ensemble_per_state[0][min_arg_SF], 0, mu, cov, cov_dets) + log_likelihood(tcs3.ensemble_per_state[0][min_arg_SF], 0, ydatas, cdinv, log_cdinv_dets, Glist)}')
    print(f'log posterior MT min: {tcs3.log_posterior_ens[1][min_arg_MT]}')
    print(f'log prior + log likelihood MT min: {log_prior(tcs3.ensemble_per_state[1][min_arg_MT], 1, mu, cov, cov_dets) + log_likelihood(tcs3.ensemble_per_state[1][min_arg_MT], 1, ydatas, cdinv, log_cdinv_dets, Glist)}')

    print('Most likely: ')
    print(tcs3.ensemble_per_state[0][max_arg_SF]) # SF MAP
    print(tcs3.ensemble_per_state[1][max_arg_MT]) # MT MAP
    print('Least likely: ')
    print(tcs3.ensemble_per_state[0][min_arg_SF]) # SF least likely
    print(tcs3.ensemble_per_state[1][min_arg_MT]) # MT least likely

    # finding the selected optimised values (map)

    SF_opt_1_map, SF_opt_2_map, SF_opt_3_map = tcs3.ensemble_per_state[0][max_arg_SF]
    MT_opt_1_map, MT_opt_2_map, MT_opt_3_map, MT_opt_4_map, MT_opt_5_map, MT_opt_6_map = tcs3.ensemble_per_state[1][max_arg_MT]

    SF_force_opt_map = ([SF_opt_1_map, SF_opt_2_map, SF_opt_3_map])*scaling_SF
    MT_force_opt_map = ([MT_opt_1_map, MT_opt_2_map, MT_opt_3_map, MT_opt_4_map, MT_opt_5_map, MT_opt_6_map])*scaling_MT

    print('SF force optimised (N): ', SF_force_opt_map)
    print('MT force optimised (N): ', MT_force_opt_map)

    SF_force_az_map = 180 - np.arctan(SF_force_opt_map[2] / SF_force_opt_map[1])
    print(SF_force_az_map)

    SF_optimised_signals_map = []
    MT_optimised_signals_map = []

    for i in range(len(stations)):
        SF_optimised_z = SF_opt_1_map*SF_e1[i][0] + SF_opt_2_map*SF_e2[i][0] + SF_opt_3_map*SF_e3[i][0]
        SF_optimised_r = SF_opt_1_map*SF_e1[i][1] + SF_opt_2_map*SF_e2[i][1] + SF_opt_3_map*SF_e3[i][1]
        SF_optimised_t = SF_opt_1_map*SF_e1[i][2] + SF_opt_2_map*SF_e2[i][2] + SF_opt_3_map*SF_e3[i][2]

        SF_optimised_signals_map.append([SF_optimised_z, SF_optimised_r, SF_optimised_t])

        MT_optimised_z = MT_opt_1_map*MT_e1[i][0] + MT_opt_2_map*MT_e2[i][0] + MT_opt_3_map*MT_e3[i][0] + MT_opt_4_map*MT_e4[i][0] + MT_opt_5_map*MT_e5[i][0] + MT_opt_6_map*MT_e6[i][0]
        MT_optimised_r = MT_opt_1_map*MT_e1[i][1] + MT_opt_2_map*MT_e2[i][1] + MT_opt_3_map*MT_e3[i][1] + MT_opt_4_map*MT_e4[i][1] + MT_opt_5_map*MT_e5[i][1] + MT_opt_6_map*MT_e6[i][1]
        MT_optimised_t = MT_opt_1_map*MT_e1[i][2] + MT_opt_2_map*MT_e2[i][2] + MT_opt_3_map*MT_e3[i][2] + MT_opt_4_map*MT_e4[i][2] + MT_opt_5_map*MT_e5[i][2] + MT_opt_6_map*MT_e6[i][2]

        MT_optimised_signals_map.append([MT_optimised_z, MT_optimised_r, MT_optimised_t])
        
    return (SF_force_opt_map, MT_force_opt_map, SF_force_az_map, SF_optimised_signals_map, MT_optimised_signals_map)


def residuals_rms(stations, ydatas, opt_dats, test_checker_x):
    residuals = []
    for i in range(len(stations)):
        station_residuals = []
        for j in range(3):  # Z, R, T components
            residual = ydatas[i][:, j] - opt_dats[i][j]
            station_residuals.append(residual)
        residuals.append(station_residuals)

    # Plot the residuals
    n_stations = len(stations)
    fig_height = 3.5 * n_stations
    fig, axes = plt.subplots(n_stations, 3, figsize=(12, fig_height), sharex=True)

    # Ensure axes is always 2D for consistency
    if n_stations == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_stations):
        for j, comp in enumerate(['Z', 'R', 'T']):
            ax = axes[i, j]
            
            # Plot the residual
            ax.plot(test_checker_x, residuals[i][j], c='purple', linewidth=2)
            
            # Add a horizontal line at zero for reference
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_ylabel(f'{stations[i]} {comp}\nResidual')
            ax.grid(True, alpha=0.3)
            
            if i == n_stations - 1:
                ax.set_xlabel('Time (s)')
            if i == 0:
                ax.set_title(f'{comp} Component Residuals')

    plt.suptitle(f'Residuals: ydata - Solutions', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print some statistics about the residuals
    print("Residual Statistics:")
    print("=" * 50)
    for i in range(len(stations)):
        print(f"Station {stations[i]}:")
        for j, comp in enumerate(['Z', 'R', 'T']):
            residual = residuals[i][j]
            rms = np.sqrt(np.mean(residual**2))
            max_abs = np.max(np.abs(residual))
            print(f"  {comp}: RMS = {rms:.2e}, Max Abs = {max_abs:.2e}")
        print()

    return residuals


## Other shit
def station_residual(d_obs, d_pred, max_shift=0, shifts_per_station=3, return_tshift=False):
    '''
    Compute the residual between observed and predicted data for a 3-component station.
    Parameters
    ----------
    d_obs : ndarray
        Observed data, shape (3, npts), shorted by Z, R, T components
    d_pred : ndarray
        Predicted data, shape (3, npts)
    max_shift : int, optional
        Maximum time shift (in samples) to consider for cross-correlation. Default is 0.
    shifts_per_station : int, optional
        Number of time shifts to compute per station. Can be 1, 2, or 3. Default is 3.
    return_tshift : bool, optional
        If True, return the list of time shifts applied. Default is False.
    Returns
    -------
    residual : ndarray
        Residual data after applying time shifts, shape (3, npts).
    tshift : list
        List of time shifts applied to each component, length 3.
    '''
    assert d_obs.shape == d_pred.shape
    assert d_obs.shape[0] == 3
    npts = d_obs.shape[1]
    if max_shift >= npts:
        raise ValueError('max_shift must be less than the number of points in the data.')
    if max_shift <= 0: # return pointwise difference if no shift is allowed
        return d_obs - d_pred, [0]*3 if return_tshift else d_obs - d_pred
    ## Compute the cross-correlation for time shifts
    f_pred = np.fft.rfft(d_pred, 2*npts-1)
    f_obs = np.fft.rfft(d_obs, 2*npts-1)
    d_xcorr = np.fft.fftshift(np.fft.irfft(f_pred * np.conj(f_obs), 2*npts-1), axes=1)
    d_xcorr = d_xcorr[:, npts-1-max_shift:npts+max_shift] # only keep relevant shifts
    t_xcorr = np.arange(-max_shift, max_shift+1)
    # t_xcorr = np.arange(-npts+1, npts)
    # print ('t_xcorr', t_xcorr)
    # print ('t_xcorr', np.arange(-npts+1, npts)[npts-1-max_shift:npts+max_shift])
    ## Determine time shifts based on shifts_per_station
    if shifts_per_station == 3: # time shift for each component independently
        tshift = [t_xcorr[np.argmax(_)] for _ in d_xcorr]
        # print ('tshift', tshift)
    elif shifts_per_station == 2: # same shift for Z and R, different for T
        tshift = [t_xcorr[np.argmax(d_xcorr[0] + d_xcorr[1])]] * 2 # same shift for Z and R
        tshift.append(t_xcorr[np.argmax(d_xcorr[2])]) # independent shift for T
    elif shifts_per_station == 1: # same shift for all components
        tshift = [t_xcorr[np.argmax(d_xcorr[0] + d_xcorr[1] + d_xcorr[2])]] * 3 # same shift for all
    else:
        return ValueError('Invalide shifts_per_station!')

    ## Apply time shifts to predicted data
    residual = np.zeros_like(d_obs)
    for i in range(3):
        # print ('obs', d_obs[i])
        # print ('pred', np.roll(d_pred[i], tshift[i]))
        residual[i] = d_obs[i] - np.roll(d_pred[i], -tshift[i])
        # print ('res', residual[i])
    if return_tshift:
        return residual, tshift
    else:
        return residual

def calc_residuals(model, model_index, Glist, ydata, num_stations=1, max_shift=50, align=True, verbose=False, returntshift=False):
    '''
    Waveform residual calculations - requires you to input the "G matrix" and returns residual waveforms for each station and component.
    '''
    r_z, r_r, r_t = [],[],[]
    p_z, p_r, p_t = [],[],[]
    for i in range(num_stations): 
        G = Glist[i][model_index] 
        data_z, data_r, data_t = ydata[i][:,0], ydata[i][:,1], ydata[i][:,2]
        model_z, model_r, model_t = np.dot(G[:,0], model), np.dot(G[:,1], model), np.dot(G[:,2], model)
        
        if(align):
            d_obs = np.vstack((data_z, data_r, data_t))
            d_pred = np.vstack((model_z, model_r, model_t))
            residuals, tshifts = station_residual(d_obs, d_pred, max_shift=max_shift, shifts_per_station=3, return_tshift=True)
            if(verbose):
                print(' Station ', i, ' offsets detected ', tshifts)
            aligned = np.zeros_like(d_pred)
            for j in range(3): aligned[j] = np.roll(d_pred[j], tshifts[j])
            model_z, model_r, model_t = aligned
        
        r_z.append(data_z - model_z)
        r_r.append(data_r - model_r)
        r_t.append(data_t - model_t)
        p_z.append(data_z - model_z)
        p_r.append(data_r - model_r)
        p_t.append(data_t - model_t)
    if(returntshift):
        return r_z,r_r,r_t, tshifts # return residual waveforms and time shifts
    else:
        return r_z,r_r,r_t # return residual waveforms

def plot_waveformes(stations, ydata, dataset2, dataset3=None, dataset4=None, dataset5=None):
    n_stations = len(stations)
    fig_height = 3.5 * n_stations  # Adjust scaling as needed
    fig, axes = plt.subplots(n_stations, 3, figsize=(12, fig_height), sharex=True)

    test_checker_x = np.arange(-150, 100, 250)

    for i in range(n_stations):
        for j, comp in enumerate(['Z', 'R', 'T']):
            ax = axes[j]
            ax.plot(test_checker_x, ydata[i][:, j], c='k', label='Data')
            ax.plot(test_checker_x, dataset2[i][j], c='b', label='Model 1')
            if dataset3 is not None:
                ax.plot(test_checker_x, dataset3[i][j], c='r', label='Model 2')
            if dataset4 is not None:
                ax.plot(test_checker_x, dataset4[i][j], c='g', label='Model 3')
            if dataset5 is not None:
                ax.plot(test_checker_x, dataset5[i][j], c='m', label='Model 4')
            ax.set_ylabel(f'{stations[i]} {comp}')
            if i == n_stations - 1:
                ax.set_xlabel('Time (s)')
            if i == 0:
                ax.set_title(comp)
            ax.grid(True)
            #ax.legend(loc='upper right', fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()
    return fig, axes


def sqExp(x1, x2, s1, rho):
    """
    Squared Exponential Covariance Function.
    This function calculates the covariance between two points.

    Args:
        x1 (float or np.ndarray): The first time point(s).
        x2 (float or np.ndarray): The second time point(s).
        s1 (float): The noise standard deviation parameter.
        rho (float): The noise correlation length parameter.

    Returns:
        float or np.ndarray: The calculated covariance.
    """
    return (s1**2) * np.exp(-(x1 - x2)**2 / (2.0 * rho**2))

def build_covariance_matrix(time, s1, rho):
    """
    Builds the covariance matrix C from the time vector and noise parameters.

    Args:
        times (np.ndarray): A 1D NumPy array of time values.
        s1 (float): The noise standard deviation parameter.
        rho (float): The noise correlation length parameter.

    Returns:
        np.ndarray: The n x n covariance matrix.
    """
    n = len(time)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = sqExp(time[i], time[j], s1, rho)
    # Add a small value to the diagonal for numerical stability (known as "nugget")
    C += np.eye(n) * 1e-6
    return C

def log_likelihood_res(params, residuals, times):
    """
    The negative log-likelihood function to be minimized.
    We minimize the negative log-likelihood because optimizers are designed for minimization.

    Args:
        params (list or np.ndarray): A list containing [s1, rho].
        residuals (np.ndarray): A 1D NumPy array of residuals (d - dp).
        times (np.ndarray): A 1D NumPy array of time values.

    Returns:
        float: The negative log-likelihood value.
    """
    import scipy.linalg as linalg

    s1, rho = params

    if s1 <= 0 or rho <= 0:
        return np.inf  # Return a large value if parameters are non-physical

    try:
        C = build_covariance_matrix(times, s1, rho)
        C_inv = linalg.inv(C)
        det_C = linalg.det(C)

        # Check for non-positive definite matrix
        if det_C <= 0:
            return np.inf

        n = len(residuals)
        
        # The log-likelihood equation
        logL = -0.5 * (n * np.log(2 * np.pi) + np.log(det_C) + residuals.T @ C_inv @ residuals)
        
        return -logL  # Return the negative log-likelihood
        
    except np.linalg.LinAlgError:
        # Handle cases where the matrix is singular or not positive definite
        return np.inf


def estimate_noise_parameters(residuals, times, s1_initial, rho_initial):
    """
    Estimates the noise parameters s1 and rho by maximizing the log-likelihood
    of the residuals.

    Args:
        residuals (np.ndarray): A 1D NumPy array of residuals (d - dp) from a best-fit model.
        times (np.ndarray): A 1D NumPy array of time values corresponding to the residuals.
        s1_initial (float): An initial guess for the s1 parameter.
        rho_initial (float): An initial guess for the rho parameter.

    Returns:
        dict: A dictionary containing the estimated s1, rho, and the final likelihood value.
    """
    initial_guess = [s1_initial, rho_initial]

    # Use a numerical optimizer to find the parameters that minimize the negative log-likelihood
    result = minimize(
        fun=log_likelihood_res,
        x0=initial_guess,
        args=(residuals, times),
        method='L-BFGS-B',  # A good choice for this type of problem
        bounds=[(1e-6, None), (1e-6, None)] # Ensure s1 and rho are positive
    )
    
    if result.success:
        s1_est, rho_est = result.x
        # The log-likelihood is the negative of the minimized function value
        logL_value = -result.fun
        return {
            "s1": s1_est,
            "rho": rho_est,
            "log_likelihood": logL_value,
            "success": True,
            "message": "Optimization successful."
        }
    else:
        return {
            "s1": None,
            "rho": None,
            "log_likelihood": None,
            "success": False,
            "message": f"Optimization failed: {result.message}"
        }

def calc_Cd_from_res(res,Cd_set,Cdinv_set,redcase):
    Cd_set_ref = Cd_set.copy() # reference data covariance matrix for case 1 (Assumes Cd_inv_set and Cd_set exist)
    Cdinv_set_ref = Cdinv_set.copy() # reference data covariance matrix for case 1 (Assumes Cd_inv_set and Cd_set exist)
    Cd_set,Cdinv_set = [],[]
    for i in range(len(Cd_set_ref)): # loop over stations
        res_z,res_r,res_t = res[0][i],res[1][i],res[2][i]
        if(redcase == 1): # solving for scale factor of Cd
            lamz = (res_z @ Cdinv_set_ref[i][0] @ res_z.T)/len(res_z)
            lamr = (res_r @ Cdinv_set_ref[i][1] @ res_r.T)/len(res_r)
            lamt = (res_t @ Cdinv_set_ref[i][2] @ res_t.T)/len(res_t)
            cdz = Cd_set_ref[i][0]*lamz
            cdr = Cd_set_ref[i][1]*lamr
            cdt = Cd_set_ref[i][2]*lamt
            cdinv_z = Cdinv_set_ref[i][0]/lamz
            cdinv_r = Cdinv_set_ref[i][1]/lamr
            cdinv_t = Cdinv_set_ref[i][2]/lamt
        else:
            cdz = np.outer(res_z, res_z.T)
            cdr = np.outer(res_r, res_r.T)
            cdt = np.outer(res_t, res_t.T)
            cdinv_z = Cd_set_ref[i][0]/(np.linalg.norm(res_z)**4)
            cdinv_r = Cd_set_ref[i][1]/(np.linalg.norm(res_r)**4)
            cdinv_t = Cd_set_ref[i][2]/(np.linalg.norm(res_t)**4)
        Cd_set.append([cdz,cdr,cdt])
        Cdinv_set.append([cdinv_z,cdinv_r,cdinv_t])
    return Cd_set,Cdinv_set,Cd_set_ref,Cdinv_set_ref