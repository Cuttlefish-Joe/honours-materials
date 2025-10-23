## useful functions for Sigs and Specs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import obspy
from obspy.clients.fdsn import Client
from matplotlib.colors import LogNorm
from obspy import Stream
from obspy.geodetics import gps2dist_azimuth

#!pip install mp_toolkits.basemap
from mpl_toolkits.basemap import Basemap #<< works with python 3.11.5

from obspy.imaging.cm import obspy_sequential
import math
from matplotlib import mlab
from matplotlib.colors import Normalize

from matplotlib.ticker import NullFormatter
from scipy.signal import hilbert

##########################################################################################

## event coords

def get_events(coords, quality=True, plot=True):
    '''
    Gets all Son catalogue events in a given region. 
    coords can be 'PIG', 'Thwaites', 'Melangé', 'Tongue' or a tuple of (lat1, lat2, lon1, lon2).
    outputs: regional latitude, longitude, time, year, magnitude lists
    '''
    if coords == 'PIG':
        lat1, lat2, lon1, lon2 = -74, -76, -97, -103
    elif coords =='Thwaites':
        lat1, lat2, lon1, lon2 = -74, -77, -106, -110
    elif coords =='Melangé':
        lat1, lat2, lon1, lon2 = -75, -75.6, -107.5, -108.5
    elif coords =='Tongue':
        lat1, lat2, lon1, lon2 = -75.1, -75.8, -106, -106.9
    else:
        lat1, lat2, lon1, lon2 = coords

    son = pd.read_csv(r"C:\Users\thele\Dropbox\PC\Desktop\Honours\python\catalogues\Son_events.csv")
    if quality:
        son = son[son['quality_group'] == 'A']
    
    lats = son['relocated_latitude'].tolist()
    lons = son['relocated_longitude'].tolist()
    mags = son['Msw'].tolist()
    times = son['relocated_time'].tolist()
    son_date_format = '%Y-%m-%dT%H:%M:%S.000000Z'
    years, months, days = [], [], []

    for i in range(len(times)):
        years.append(int(datetime.strptime(times[i], son_date_format).strftime('%Y')))
        months.append(int(datetime.strptime(times[i], son_date_format).strftime('%m')))
        days.append(int(datetime.strptime(times[i], son_date_format).strftime('%d')))
        
    # regional sorting
    reg_lats, reg_lons, reg_mags, reg_times, reg_years = [], [], [], [], []
    for i in range(len(lats)):
        if lats[i] <= lat1 and lats[i] >= lat2 and lons[i] <= lon1 and lons[i] >= lon2:
            reg_lats.append(lats[i])
            reg_lons.append(lons[i])
            reg_mags.append(mags[i])
            reg_times.append(times[i])
            reg_years.append(years[i] + months[i]/12 + days[i]/365)
    print('Number of events in region:', len(reg_lats))
    
    if plot:
        urcrlonc, urcrnrlatc = lon1, lat2 # upper right corner
        llcrlonc, llcrnrlatc = lon2, lat1 # lower left corner

        fig = plt.figure(figsize=(9, 6))
        xa = fig.add_axes([0.90, 0.90, 0.90, 0.90])
        ma = Basemap(epsg='3031', llcrnrlon=llcrlonc, llcrnrlat=llcrnrlatc, urcrnrlon=urcrlonc, urcrnrlat=urcrnrlatc, ax=xa)
        wa, ha = ma.xmax, ma.ymax
        ma.drawmapboundary(fill_color='aqua')
        ma.fillcontinents(color='ivory', lake_color='aqua')
        son = ma.scatter(reg_lons, reg_lats, c=reg_mags, s=20, cmap='autumn', vmin=3.1, vmax=3.7, latlon=True, label='Pham (2025, in prep.)')
        plt.colorbar(son, label='Magnitude')
        plt.show()

    return reg_lats, reg_lons, reg_times, reg_years, reg_mags


## stations

def get_stations(starttime=obspy.UTCDateTime(2010,1,1,1), endtime=obspy.UTCDateTime(2024,1,1,1)):
    """
    Get stations from IRIS for a given time period (Antarctica).
    outputs: station latitudes, station longitudes, flat station codes, flat network codes, start time, end time, inventory
    """
    ap = Client('IRIS')
    inv_LH = ap.get_stations(network='*', station="*", level='response', channel='LHZ', starttime=starttime,
                              endtime=endtime, maxlatitude=-63)
    sta_lats = []
    sta_lons = []
    for i in range(len(inv_LH)):
        for j in range(len(inv_LH[i])):
            sta_lats.append(inv_LH[i][j].latitude)
            sta_lons.append(inv_LH[i][j].longitude)
    
    print('Number of stations:', len(sta_lats))
    
    stations_flat = []
    networks_flat = []
    for i in range(len(inv_LH)):
        for j in range(len(inv_LH[i])):
            stations_flat.append(inv_LH[i][j].code)
            networks_flat.append(inv_LH[i].code)

    # sort out creation and termination dates for each station
    obspy_date_format = '%Y-%m-%dT%H:%M:%S.%fZ'

    station_starts = []
    station_ends = []

    for i in range(len(inv_LH)):
        for j in range(len(inv_LH[i])):
            
            startyear = int(datetime.strptime(str(inv_LH[i][j].start_date), obspy_date_format).strftime('%Y'))
            startmonth = int(datetime.strptime(str(inv_LH[i][j].start_date), obspy_date_format).strftime('%m'))
            startday = int(datetime.strptime(str(inv_LH[i][j].start_date), obspy_date_format).strftime('%d'))
            
            if inv_LH[i][j].end_date == None:
                endyear = 2025
                endmonth = 1
                endday = 1
            else:
                endyear = int(datetime.strptime(str(inv_LH[i][j].end_date), obspy_date_format).strftime('%Y'))
                endmonth = int(datetime.strptime(str(inv_LH[i][j].end_date), obspy_date_format).strftime('%m'))
                endday = int(datetime.strptime(str(inv_LH[i][j].end_date), obspy_date_format).strftime('%d'))
            
            station_starts.append(startyear + startmonth/12 + startday/365)
            station_ends.append(endyear + endmonth/12 + endday/365)
            
    return sta_lats, sta_lons, stations_flat, networks_flat, station_starts, station_ends, inv_LH


## trace searcher

def distance(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0]

def trace_searcher(start_distance, end_distance, event_info, station_info, channel='LHZ', timebuff=400):
    '''
    find traces between given distances for given station and event catalogues. 
    outputs: the traces, the event numbers, the arrival times, and the station numbers
    '''
    ap = Client('IRIS')

    loc_lats, loc_lons, loc_times, loc_years, loc_mags = event_info
    sta_lats, sta_lons, stations_flat, networks_flat, station_starts, station_ends, inv = station_info

    signals = Stream()
    arrival_times = []
    station_numbers = []
    event_numbers = []
    
    for i in range(len(sta_lats)):
        for j in range(len(loc_lats)):
            dist = distance(sta_lats[i], sta_lons[i], loc_lats[j], loc_lons[j]) / 1000 # in km
            if dist >= start_distance and dist < end_distance:
                event_time = loc_years[j]
                if event_time >= station_starts[i] and event_time <= station_ends[i]:
                    print('Event:', j, 'Station:', i, 'Distance:', dist)
                    try:
                        time_to_arrive = dist / 3 # assuming a wave speed of 3 km/s
                        begin_time = obspy.UTCDateTime(loc_times[j]) + time_to_arrive - timebuff # 450 second period recording around 
                        end_time = obspy.UTCDateTime(loc_times[j]) + time_to_arrive + timebuff # the theoretical arrival time (3km/s speed)
                        st = ap.get_waveforms(network=networks_flat[i], station=stations_flat[i], location='*', channel=channel, starttime=begin_time, endtime=end_time)
                        st.remove_response(inv,output='DISP',taper=True, taper_fraction=0.1)
                        st.detrend('linear')
                        st.detrend('demean')
                        st.taper(max_percentage=0.1) # original 0.1
                        signals += st
                        arrival_times.append(obspy.UTCDateTime(loc_times[j]) + time_to_arrive)
                        station_numbers.append(i) 
                        event_numbers.append(j)
                    except Exception as ex:
                        print(ex)
                        
    return signals, event_numbers, arrival_times, station_numbers


## runner

def runner(event_info, station_info, 
           start_distances=[150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
           end_distances=[200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], channel='LHZ', timebuff=400):
    '''
    finds the traces for the given event and station catalogues, plus distance bands. 
    outputs: the traces, the event numbers, the arrival times, and the station numbers for each distance band
    '''
    if len(start_distances) != len(end_distances):
        raise ValueError("start_distances and end_distances must have the same length.")

    num_distances = len(start_distances)
    loc_lats, loc_lons, loc_times, loc_years, loc_mags = event_info
    sta_lats, sta_lons, stations_flat, networks_flat, station_starts, station_ends, inv = station_info

    t_distance_streams = []
    t_event_numbers_streams = []
    t_arrival_times_streams = []
    t_station_numbers_streams = []

    for i in range(num_distances):
        Thwaites, t_num, t_arrival_times, t_station_num = trace_searcher(
            start_distances[i], end_distances[i], event_info=event_info, station_info=station_info, channel=channel, timebuff=timebuff)
        t_distance_streams.append(Thwaites)
        t_event_numbers_streams.append(t_num)
        t_arrival_times_streams.append(t_arrival_times)
        t_station_numbers_streams.append(t_station_num)
    
    return t_distance_streams, t_event_numbers_streams, t_arrival_times_streams, t_station_numbers_streams

## filterer

def set_filterer(stream):
    stream_10_100 = stream.copy()
    stream_10_100.filter("bandpass", freqmin=0.01, freqmax=0.1, corners=3, zerophase=True)

    stream_10_20 = stream.copy()
    stream_10_20.filter("bandpass", freqmin=0.05, freqmax=0.1, corners=3, zerophase=True)

    stream_10_30 = stream.copy()
    stream_10_30.filter("bandpass", freqmin=0.03, freqmax=0.1, corners=3, zerophase=True)

    stream_10_50 = stream.copy()
    stream_10_50.filter("bandpass", freqmin=0.02, freqmax=0.1, corners=3, zerophase=True)

    stream_17_25 = stream.copy()
    stream_17_25.filter("bandpass", freqmin=0.04, freqmax=0.06, corners=3, zerophase=True)

    stream_40_70 = stream.copy()
    stream_40_70.filter("bandpass", freqmin=0.015, freqmax=0.025, corners=3, zerophase=True)

    stream_50_100 = stream.copy()
    stream_50_100.filter("bandpass", freqmin=0.01, freqmax=0.02, corners=3, zerophase=True)
    
    stream_50_150 = stream.copy()
    stream_50_150.filter("bandpass", freqmin=0.006, freqmax=0.02, corners=3, zerophase=True)

    stream_1_10 = stream.copy()
    stream_1_10.filter("bandpass", freqmin=0.1, freqmax=1.0, corners=3, zerophase=True)
    
    stream_01_10 = stream.copy()
    stream_01_10.filter("bandpass", freqmin=0.1, freqmax=10, corners=3, zerophase=True)
    
    stream_5_10 = stream.copy()
    stream_5_10.filter("bandpass", freqmin=0.1, freqmax=0.2, corners=3, zerophase=True)

    return [stream, stream_10_100, stream_10_20, stream_10_30, stream_10_50,
            stream_17_25, stream_40_70, stream_50_100, stream_50_150, stream_1_10,
            stream_01_10, stream_5_10]


## outputter

def outputter(streams, trim_amount=200):
    '''
    filters and trims all the streams to get the edges stuff under control. 
    outputs: filtered distance streams ([number of distances][number of traces][traces])
    '''
    num_distances = len(streams)
    
    filtered_streams = []
    for i in range(num_distances):
        filtered_streams.append(set_filterer(streams[i]))
    
    for i in range(num_distances):
        for j in range(len(filtered_streams[i])):
            for k in range(len(filtered_streams[i][j])):
                filtered_streams[i][j][k].trim(starttime=filtered_streams[i][j][k].stats.starttime + trim_amount, endtime=filtered_streams[i][j][k].stats.endtime - trim_amount)
    
    distance_streams_filtered = filtered_streams

    return distance_streams_filtered

def sets_frequency(streams, frequency=1):
    """
    Returns a list of streams filtered in position frequency (default 10-100s)
    """
    return [stream[frequency] for stream in streams]


## plotting

def plot_and_quality(traces, arrivals=None, events=None, stations=None, frequency=None, minval=None, maxval=None, centrebar=200): # quality part needs fixing
    '''
    plot the waveforms and filter them between minval and maxval, to get quality data sets
    '''
    colours = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink', 'gray', 'olive', 'brown', 'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime', 'coral']
    distance_bands = ['150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500', '500-550', '550-600', '600-650', '650-700', '700-750', '750-800', '800-850', '850-900', '900-950', '950-1000']

    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1)

    if minval is None:
        for i in range(len(traces)):
            for j in range(len(traces[i])):
                if j == 0:
                    ax.plot(traces[i][j].times(), traces[i][j].data, label=f'{distance_bands[i]} km', color=colours[i])
                else:
                    ax.plot(traces[i][j].times(), traces[i][j].data, color=colours[i])
    else:
        for i in range(len(traces)):
            for j in range(len(traces[i])):
                if max(abs(traces[i][j].data)) >= minval and max(abs(traces[i][j].data)) <= maxval:
                    if j == 0:
                        ax.plot(traces[i][j].times(), traces[i][j].data, label=f'{distance_bands[i]} km', color=colours[i])
                    else:
                        ax.plot(traces[i][j].times(), traces[i][j].data, color=colours[i])

    ax.axvline(centrebar, 0, 1, color='k', linewidth=2) # vertical line at the approximate arrival time (same for all, relatively)
    ax.set_title('Events by frequency band at stations', fontsize=17, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (m/s)')

    # Place a single legend for the whole figure
    fig.legend(loc='upper right', bbox_to_anchor=(1, 0.885), frameon=True)
    plt.show()

    # # create quality sets
    # sets_quality = [
    #     [tr for tr in sublist if minval is None or (minval <= max(abs(tr.data)) <= maxval)]
    #     for sublist in traces
    # ]
    # arrivals_quality = [
    #     [tr for tr in sublist if minval is None or (minval <= max(abs(tr.data)) <= maxval)]
    #     for sublist in arrivals
    # ]
    # events_quality = [
    #     [tr for tr in sublist if minval is None or (minval <= max(abs(tr.data)) <= maxval)]
    #     for sublist in events
    # ]
    # stations_quality = [
    #     [tr for tr in sublist if minval is None or (minval <= max(abs(tr.data)) <= maxval)]
    #     for sublist in stations
    # ]
    #return sets_quality, arrivals_quality, events_quality, stations_quality
    return fig, ax

def event_sorter(traces, events, stations, num_events):
    '''
    Sorts the events into separate lists based on their event number.
    '''
    t_events = [[] for _ in range(num_events)]
    t_stations = [[] for _ in range(num_events)]
    for i in range(len(traces)):
        for j in range(len(traces[i])):
            event_num = events[i][j]
            if 0 <= event_num < num_events:
                t_events[event_num].append(traces[i][j])
                t_stations[event_num].append(stations[i][j])
                
    return t_events, t_stations

## spectrograms

def single_spectrogram(trace):
    fs = 1.0 # sampling rate in Hz
    nfft = 128  # number of points in FFT
    pad_to = 1024  # padding to next power of 2
    noverlap = 115 # number of points to overlap between segments
    vmin, vmax = 0.0, 1.0  # color scale limits
    
    if len(trace.data) < 300:
        return None
    
    spec, freq, time = mlab.specgram(trace.data, Fs=fs, NFFT=nfft, pad_to=pad_to, noverlap=noverlap)
    spec = np.sqrt(spec[1:, :])
    freq = freq[1:]
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
    time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
    time -= halfbin_time
    freq -= halfbin_freq
    # normalising the spectrogram
    vmin = spec.min() + vmin * float(spec.max() - spec.min())
    vmax = spec.min() + vmax * float(spec.max() - spec.min())
    norm = Normalize(vmin, vmax, clip=True)
    
    data_gouraud = np.pad(spec, ((0, len(freq) - spec.shape[0]), (0, len(time) - spec.shape[1])), mode='edge')
    
    return time, freq, data_gouraud, norm


def avg_spectrogram(traces):
    '''
    standard averaging spectrogram
    '''
    fs = 1.0 # sampling rate in Hz
    nfft = 128  # number of points in FFT
    pad_to = 1024  # padding to next power of 2
    noverlap = 115 # number of points to overlap between segments
    vmin, vmax = 0.0, 1.0  # color scale limits
    # extracting the data
    datas = []
    for trace in traces:
        if len(trace.data) > 300: # somewhat arbitrary - we expect 400 and can't use 128 or below lengths
            datas.append(trace.data)
    # doing the spectrogram calculation
    
    if len(datas) == 0:
        return None  # no data to process
    else:
        specs, freqs, times = [], [], []
        for data in datas:
            spec, freq, time = mlab.specgram(data, Fs=fs, NFFT=nfft, pad_to=pad_to, noverlap=noverlap)
            spec = np.sqrt(spec[1:, :])
            freq = freq[1:]
            halfbin_time = (time[1] - time[0]) / 2.0
            halfbin_freq = (freq[1] - freq[0]) / 2.0
            freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
            time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
            time -= halfbin_time
            freq -= halfbin_freq
            specs.append(spec)
            freqs.append(freq)
            times.append(time)
        # making them all of equal size
        min_time_bins = (min(specs[i].shape[1] for i in range(len(specs))))
        min_freq_bins = (min(specs[i].shape[0] for i in range(len(specs))))
        for i in range(len(specs)):
            specs[i] = specs[i][:min_freq_bins, :min_time_bins]
        # averaging the spectrograms
        avg_spec = np.mean(specs, axis=0)
        # normalising the spectrogram
        vmin = avg_spec.min() + vmin * float(avg_spec.max() - avg_spec.min())
        vmax = avg_spec.min() + vmax * float(avg_spec.max() - avg_spec.min())
        norm = Normalize(vmin, vmax, clip=True)

        return times[0][:min_time_bins], freqs[0][:min_freq_bins], avg_spec, norm
    
def avg_mean_spectrogram(traces):
    '''
    normalised averaging spectrogram
    '''
    fs = 1.0 # sampling rate in Hz
    nfft = 128  # number of points in FFT
    pad_to = 1024  # padding to next power of 2
    noverlap = 115 # number of points to overlap between segments
    vmin, vmax = 0.0, 1.0  # color scale limits
    # extracting the data
    datas = []
    for trace in traces:
        if len(trace.data) > 300: # somewhat arbitrary - we expect 400 and can't use 128 or below lengths
            datas.append(trace.data)
    # doing the spectrogram calculation
    
    if len(datas) == 0:
        return None  # no data to process
    else:
        specs, freqs, times = [], [], []
        for data in datas:
            spec, freq, time = mlab.specgram(data, Fs=fs, NFFT=nfft, pad_to=pad_to, noverlap=noverlap)
            spec = np.sqrt(spec[1:, :])
            freq = freq[1:]
            halfbin_time = (time[1] - time[0]) / 2.0
            halfbin_freq = (freq[1] - freq[0]) / 2.0
            freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
            time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
            time -= halfbin_time
            freq -= halfbin_freq
            specs.append(spec)
            freqs.append(freq)
            times.append(time)
        # making them all of equal size
        min_time_bins = (min(specs[i].shape[1] for i in range(len(specs))))
        min_freq_bins = (min(specs[i].shape[0] for i in range(len(specs))))
        for i in range(len(specs)):
            specs[i] = specs[i][:min_freq_bins, :min_time_bins]
            vmin = specs[i].min() + vmin * float(specs[i].max() - specs[i].min())
            vmax = specs[i].min() + vmax * float(specs[i].max() - specs[i].min())
            specs[i] = (specs[i] - specs[i].min()) / (specs[i].max() - specs[i].min())  # normalising each spectrogram
            
        # averaging the spectrograms
        avg_spec = np.mean(specs, axis=0)

        return times[0][:min_time_bins], freqs[0][:min_freq_bins], avg_spec
    
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]
    
def plot_fancy(traces, left=0.1, bottom=0.1, h_1=0.2, h_2=0.6, w_1=0.2, w_2=0.6):
    '''
    standard fancy plot of spectrogram, signal, and FFT - FFT is not normalised
    '''
    ## spectrogram bit
    fs = 1.0 # sampling rate in Hz
    nfft = 128  # number of points in FFT
    pad_to = 1024  # padding to next power of 2
    noverlap = 115 # number of points to overlap between segments
    vmin, vmax = 0.0, 1.0  # color scale limits
    # extracting the data
    datas = []
    datas_times = []
    for trace in traces:
        if len(trace.data) > 300: # somewhat arbitrary - we expect 400 and can't use 128 or below lengths
            datas.append(trace.data)
            datas_times.append(trace.times())
            
    datas_max = max([max(data) for data in datas])
    datas_min = min([min(data) for data in datas])
    
    # Hilbert envelopes
    envelopes = []
    for i in range(len(datas)):
        analytic_signal = hilbert(datas[i])
        envelope = np.abs(analytic_signal)
        envelopes.append(envelope)
    
    # average signal
    average_signal = np.mean(envelopes, axis=0)
    
    # doing the spectrogram calculation
    
    if len(datas) == 0:
        return None  # no data to process
    else:
        specs, freqs, times = [], [], []
        for data in datas:
            spec, freq, time = mlab.specgram(data, Fs=fs, NFFT=nfft, pad_to=pad_to, noverlap=noverlap)
            spec = np.sqrt(spec[1:, :])
            freq = freq[1:]
            halfbin_time = (time[1] - time[0]) / 2.0
            halfbin_freq = (freq[1] - freq[0]) / 2.0
            freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
            time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
            time -= halfbin_time
            freq -= halfbin_freq
            specs.append(spec)
            freqs.append(freq)
            times.append(time)
        # making them all of equal size
        min_time_bins = (min(specs[i].shape[1] for i in range(len(specs))))
        min_freq_bins = (min(specs[i].shape[0] for i in range(len(specs))))
        for i in range(len(specs)):
            specs[i] = specs[i][:min_freq_bins, :min_time_bins]
            vmin = specs[i].min() + vmin * float(specs[i].max() - specs[i].min())
            vmax = specs[i].min() + vmax * float(specs[i].max() - specs[i].min())
            specs[i] = (specs[i] - specs[i].min()) / (specs[i].max() - specs[i].min())  # normalising each spectrogram
            
        # averaging the spectrograms
        avg_spec = np.mean(specs, axis=0)

        spectrogram_data = times[0][:min_time_bins], freqs[0][:min_freq_bins], avg_spec
        
    ## FFT bit
    spectrums = []
    ns = []
    fft_datas = []
    for i in range(len(datas)):
        fft = np.fft.fft(datas[i])
        fft_datas.append(fft)
        nn = len(datas[i])
        ns.append(nn)
        spectrums.append(np.fft.fftfreq(nn, 1))
    
    fft_average = np.mean(np.abs(fft_datas), axis=0)
    
    ## plotting    
    fig = plt.figure(figsize=(7, 7))
    # plot signals
    ax_sig = fig.add_axes([left + w_1, bottom, w_2, h_1])
    for i in range(len(datas)):
        ax_sig.plot(datas_times[i], datas[i], color='grey')
    ax_sig.plot(datas_times[0], average_signal, color='C0', linewidth=2, label='average signal')
    ax_sig.plot(datas_times[0][np.argmax(average_signal)], np.max(average_signal), 'or')
    ax_sig.axvline(200, 0, 1, color='k', linewidth=2) # vertical line at the approximate arrival time
    # plot spectrogram
    ax_spec = fig.add_axes([left + w_1, bottom + h_1, w_2, h_2])
    x, y, data = spectrogram_data
    ax_spec.set_yscale('log')  # Set y-axis to log scale
    img_spec = ax_spec.pcolormesh(x, y, data, shading='gouraud')
    img_spec.set_rasterized(True)
    ax_spec.axvline(200, 0, 1, color='k', linewidth=2) # vertical line at the approximate arrival time
    ax_spec.axhline(0.04, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)
    ax_spec.axhline(0.06, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)
    # plot fft
    ax_fft = fig.add_axes([left, bottom + h_1, w_1, h_2])
    for i in range(len(spectrums)):
        ax_fft.semilogy(np.abs(fft_datas[i][:ns[i] // 2]), spectrums[i][:ns[i] // 2], color='grey')
    ax_fft.semilogy(np.abs(fft_average[:ns[0] // 2]), spectrums[0][:ns[0] // 2], color='C0', linewidth=2, label='average FFT')
    
    ax_fft.plot(np.max(fft_average), spectrums[0][np.argmax(fft_average[:ns[0] // 2])], 'or', label='peak FFT')
    
    #ax_fft.plot(datas_times[0][np.argmax(average_signal)], np.max(average_signal), 'or')

    ax_fft.axhline(0.04, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)
    ax_fft.axhline(0.06, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)
    
    ax_spec.set_ylim(0.009, 0.2)
    ax_spec.set_xlim(x[0], x[-1])
    ax_sig.set_ylim(datas_min * 1.1, datas_max * 1.1)
    ax_sig.set_xlim(x[0], x[-1])
    #xlim = spectrums_max * 1.1
    #ax_fft.set_xlim(xlim, 0.)
    ax_fft.set_ylim(0.009, 0.2)
    ax_fft.invert_xaxis()  # Invert x-axis for FFT plot
    

    ax_sig.set_xlabel('time')
    ax_fft.set_ylabel('frequency')

    # remove axis labels
    ax_spec.xaxis.set_major_formatter(NullFormatter())
    ax_spec.yaxis.set_major_formatter(NullFormatter())
    ax_fft.xaxis.set_major_formatter(NullFormatter())
    
    plot_title = namestr(traces, globals())
    
    plt.suptitle(f'{plot_title[0]} - Spectrogram, Signal and FFT', fontsize=16, fontweight='bold')

    plt.show()
    
def plot_fancy_norm(traces, name=None, savefig=False, left=0.1, bottom=0.1, h_1=0.2, h_2=0.6, w_1=0.2, w_2=0.6):
    '''
    plot of spectrogram, signal, and FFT, with normalised and unnormalised averages in place
    '''
    ## spectrogram bit
    fs = 1.0  # sampling rate in Hz
    nfft = 128  # number of points in FFT
    pad_to = 1024  # padding to next power of 2
    noverlap = 115  # number of points to overlap between segments
    vmin, vmax = 0.0, 1.0  # color scale limits

    # extracting the data
    datas = []
    datas_times = []
    for trace in traces:
        if len(trace.data) > 300:  # somewhat arbitrary - we expect 400 and can't use 128 or below lengths
            datas.append(trace.data)
            datas_times.append(trace.times())

    datas_max = max([max(data) for data in datas])
    datas_min = min([min(data) for data in datas])

    # Hilbert envelopes
    envelopes = []
    envelopes_max = []
    envelopes_norm = []
    for i in range(len(datas)):
        analytic_signal = hilbert(datas[i])
        envelope = np.abs(analytic_signal)
        envelopes_max.append(np.max(envelope))
        envelope_norm = envelope / np.max(envelope)  # normalising the envelope
        envelopes.append(envelope)
        envelopes_norm.append(envelope_norm)

    # average signal
    average_signal = np.mean(envelopes, axis=0)
    average_signal_norm = np.mean(envelopes_norm, axis=0) * np.max(envelopes_max)

    # doing the spectrogram calculation
    if len(datas) == 0:
        return None  # no data to process
    else:
        specs, freqs, times = [], [], []
        for data in datas:
            spec, freq, time = mlab.specgram(data, Fs=fs, NFFT=nfft, pad_to=pad_to, noverlap=noverlap)
            spec = np.sqrt(spec[1:, :])
            freq = freq[1:]
            halfbin_time = (time[1] - time[0]) / 2.0
            halfbin_freq = (freq[1] - freq[0]) / 2.0
            freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
            time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
            time -= halfbin_time
            freq -= halfbin_freq
            specs.append(spec)
            freqs.append(freq)
            times.append(time)

        # making them all of equal size
        min_time_bins = min(spec.shape[1] for spec in specs)
        min_freq_bins = min(spec.shape[0] for spec in specs)
        for i in range(len(specs)):
            specs[i] = specs[i][:min_freq_bins, :min_time_bins]
            vmin = specs[i].min() + vmin * float(specs[i].max() - specs[i].min())
            vmax = specs[i].min() + vmax * float(specs[i].max() - specs[i].min())
            specs[i] = (specs[i] - specs[i].min()) / (specs[i].max() - specs[i].min())  # normalize

        avg_spec = np.mean(specs, axis=0)
        spectrogram_data = times[0][:min_time_bins], freqs[0][:min_freq_bins], avg_spec

    ## FFT bit
    spectrums = []
    ns = []
    fft_datas = []
    fft_datas_norm = []
    fft_max = []
    for i in range(len(datas)):
        fft = np.fft.fft(datas[i])
        fft_max.append(np.max(np.abs(fft)))
        fft_norm = fft / np.max(np.abs(fft))  # normalize
        fft_datas.append(fft)
        fft_datas_norm.append(fft_norm)
        nn = len(datas[i])
        ns.append(nn)
        spectrums.append(np.fft.fftfreq(nn, 1))

    fft_average = np.mean(np.abs(fft_datas), axis=0)
    fft_average_norm = np.mean(np.abs(fft_datas_norm), axis=0) * np.max(fft_max)

    ## plotting
    fig = plt.figure(figsize=(7, 7))

    # plot signals
    ax_sig = fig.add_axes([left + w_1, bottom, w_2, h_1])
    for i in range(len(datas)):
        ax_sig.plot(datas_times[i], datas[i], color='grey')
    ax_sig.plot(datas_times[0], average_signal, color='C0', linewidth=2, label='avg signal')
    ax_sig.plot(datas_times[0], average_signal_norm, color='C1', linewidth=2, label='avg norm signal')
    ax_sig.plot(datas_times[0][np.argmax(average_signal)], np.max(average_signal), 'or')
    ax_sig.plot(datas_times[0][np.argmax(average_signal_norm)], np.max(average_signal_norm), 'or')
    ax_sig.axvline(200, 0, 1, color='k', linewidth=2, alpha=0.7)

    # plot spectrogram
    ax_spec = fig.add_axes([left + w_1, bottom + h_1, w_2, h_2])
    x, y, data = spectrogram_data
    ax_spec.set_yscale('log')
    img_spec = ax_spec.pcolormesh(x, y, data, shading='gouraud')
    img_spec.set_rasterized(True)
    ax_spec.axvline(200, 0, 1, color='k', alpha=0.7, linewidth=2, label='expected arrival')
    ax_spec.axhline(0.04, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2, label='expected freq.')
    ax_spec.axhline(0.06, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)

    # plot fft
    ax_fft = fig.add_axes([left, bottom + h_1, w_1, h_2])
    for i in range(len(spectrums)):
        ax_fft.semilogy(np.abs(fft_datas[i][:ns[i] // 2]), spectrums[i][:ns[i] // 2], color='grey')
    ax_fft.semilogy(np.abs(fft_average[:ns[0] // 2]), spectrums[0][:ns[0] // 2], color='C0', linewidth=2, label='average')
    ax_fft.semilogy(np.abs(fft_average_norm[:ns[0] // 2]), spectrums[0][:ns[0] // 2], color='C1', linewidth=2, label='norm. average\n(scaled)')
    ax_fft.plot(np.max(fft_average), spectrums[0][np.argmax(fft_average[:ns[0] // 2])], 'or')
    ax_fft.plot(np.max(fft_average_norm), spectrums[0][np.argmax(fft_average_norm[:ns[0] // 2])], 'or')
    ax_fft.axhline(0.04, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)
    ax_fft.axhline(0.06, 0, 1, color='skyblue', alpha=0.5, ls='--', linewidth=2)

    #ax_fft.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 20.0) * 0.1, numticks=15))
    
    ax_spec.set_ylim(0.009, 0.2)
    ax_spec.set_xlim(x[0], x[-1])
    ax_sig.set_ylim(datas_min * 1.1, datas_max * 1.1)
    ax_sig.set_xlim(x[0], x[-1])
    ax_fft.set_ylim(0.009, 0.2)
    ax_fft.invert_xaxis()

    ax_sig.set_xlabel('time (s)')
    ax_fft.set_ylabel('frequency (Hz)')

    # remove axis labels
    ax_spec.xaxis.set_major_formatter(NullFormatter())
    ax_spec.yaxis.set_major_formatter(NullFormatter())
    ax_fft.xaxis.set_major_formatter(NullFormatter())

    # plot titling
    plot_title = namestr(traces, globals())
    if name is not None:
        plt.suptitle(f'{name}', fontsize=16, fontweight='bold')
    elif plot_title:
        plt.suptitle(f'{plot_title[0]} - Spectrogram, Signal and FFT', fontsize=16, fontweight='bold')    
    else:
        plt.suptitle('Spectrogram, Signal and FFT', fontsize=16, fontweight='bold')


    plt.legend(bbox_to_anchor=(0.85, -0.1, 0.02, 0.01))
    
    if savefig is True:
        plt.savefig(f'{name}.png', bbox_inches='tight', dpi=300)

    #plt.suptitle(f'{plot_title[0]} - Spectrogram, Signal and FFT', fontsize=16, fontweight='bold')
    plt.show()
    return plot_title
    
## joy division plot

## station identification and distance from events (create a list) - names list and distances list

## 


