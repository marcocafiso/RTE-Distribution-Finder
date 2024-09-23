import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, sosfilt
import pandas as pd
import scipy.ndimage.filters as ndif
import matplotlib.pyplot as plt
import math
# import matplotlib.ticker as ticker
# from ast import literal_eval
from MFDFA import MFDFA
import os
# import time as tm
import sys
import pwlf
from PIL import Image


###################################################################################################################
############################################## USEFUL FUNCTIONS ###################################################
###################################################################################################################
def fig2img(fig):
    """
    Convert Matplotlib figure to PIL Image

    Args:
        fig (matplotlib.figure): Matplotlib figure

    Returns:
        img : img figure
    """
    
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    
    return Image.frombytes("RGB", (width, height), buf)

def display_progress(start_time, state, idx, end_idx):
    """
    Display progress of simulation every 5%.
    Include a progress bar and a time estimate.

    Parameters
    ----------
    epoch : int
        Number of epoch.
    start_time : float
        Time at the start of the simulation.
    state : float
        Initial time of the simulation. 
    idx : int
        Current time index.
    end_idx : int
        End time index.

    Returns
    -------
    None.

    """

    progress = (idx+1)/end_idx

    if progress >= 0.9:
        progress = 1.0

    if progress - state >= 0.05:  # >= state - 1e-6:
        state = progress
        elapsed_time = tm.time() - start_time
        estimated_time = np.abs(elapsed_time/progress - elapsed_time)
        # format time as hh:mm:ss
        elapsed_time = tm.strftime("%H:%M:%S", tm.gmtime(elapsed_time))
        estimated_time = tm.strftime(
        "%H:%M:%S", tm.gmtime(estimated_time))
        # create progress bar
        block = int(round(20*progress))
        # text = f"\rEpoch {epoch + 1} - Progress: [{'#' * block + '-' * (20 - block)}] {progress * 100:.0f}%"
        text = f"\rProgress: [{'#' * block + '-' * (20 - block)}] {progress * 100:.0f}%"
        # add elapsed time to text
        text += f" - {elapsed_time} elapsed"
        # add estimated time to text
        text += f" - {estimated_time} remaining" + 5*" "
        # flush previous text and print new texts
        sys.stdout.write(text)
        sys.stdout.flush()

    return



###############################################################################################################################################################################
############################################################################# EVENT DETECTION #################################################################################
###############################################################################################################################################################################

def RTP(signal, sfreq, N=None, percentile=90, n=5, level_signal='True'):
    """
    This function computes the RTP events from a signal timeseries. 
    Two step are required:
        1. Detection of preliminary events
        2. Selection of true RTPs events from those found at point 1.
    Input:
        - signal: signal timeseries
        - sfreq: sampling frequency of signal
        - N: window length for signal filtering
        - percentile: threshold for selecting true events
    Output:
        - true_RTP: selected RTP events
        - TS: Test Signal
        - LS: Level Signal
        - derivative: Derivative of TS
    """

    def get_index_positions_by_condition(list_of_elems, condition):
        """ 
        Returns the indexes of items in the list that satisfy condition().
        """
        index_pos_list = []
        for i in range(len(list_of_elems)):
            if condition(list_of_elems[i]) == True:
                index_pos_list.append(i)
        return index_pos_list

    def preliminary_RTP(signal, N):
        """
        This function implements preliminary detection (step 1). 
        First the test signal (TS) sequence is obtained from the modulus of the hilbert transform of the signal. 
        Then the level signal (LS) sequence is derived applying a moving average on TS.
        From the intersections between TS and LS the time locations of the events is extracted.

        """
        # Computing the envelope of the signal
        TS = np.abs(hilbert(signal))   # tolgo parte reale rispetto codice L.

        if N == None:  # se non metto nulla python dà errore
            # lo lascio nel caso dovesse servire media su tutta serie temporale
            N = len(TS)

        if level_signal == 'True':

            # Smoothing TS with a moving average of lenght N
            # considero come se ci fossero condizioni periodiche al bordo
            LS = ndif.uniform_filter1d(TS, N, mode='wrap', origin=0)

            # Finding intersections between signals as the indexes corrisponding to sign changes between two consecutive elements
            # in the array TS - LS
            if np.isnan(TS).any() == True:
                pre_RTP = []
            else:
                pre_RTP = np.argwhere(np.diff(np.sign(TS - LS))).flatten()

            return pre_RTP, TS

        else:
            return TS

    def get_indexes_true_RTP(TS, percentile_value, n):
        """
        This function looks for statistically significant RTPs (step 2).

        From the signal (TS - LS or only TS) we estimated is derivative for each
        time-step by using the previous and next 12 points. 
        After we extract from the distribution of the derivative only the time-steps
        regarding the desired percentile value of the distribution.

        """
        S = TS
        derivative = []
        step_size = 1/sfreq  # frequenza di campionamento per calcolare step

        # Now i have to calculate up to the fifth value around a pre_RTP. Thus now I calculate this derivative on the S
        # obtained above (calcola derivata facendo la media delle pendenze tra punti sequenzialmente e simmetricamente più lontani da pre_RTP(fino a 12) )
        if len(S) > 0:

            # Derivative for extreme left element
            right_derivative = np.divide(S[1]-S[0], step_size)
            derivative.append(right_derivative)

            for el in range(1, n):
                sum_array = [np.divide(S[el+i]-S[el-i], (2*i)*step_size)
                             for i in range(1, el+1)]
                derivative.append(1/len(sum_array) * np.sum(sum_array))

            for el in range(n, len(S)-n):
                sum_array = [np.divide(S[el+i]-S[el-i], (2*i)*step_size)
                             for i in range(1, n+1)]
                derivative.append(1/len(sum_array) * np.sum(sum_array))

            for el in range(len(S)-n, len(S)-1):
                sum_array = [np.divide(S[el+i]-S[el-i], (2*i)*step_size)
                             for i in range(1, len(S)-el)]
                derivative.append(1/len(sum_array) * np.sum(sum_array))

            # Derivative for extreme right element
            left_derivative = np.divide(S[len(S)-1]-S[len(S)-2], step_size)
            derivative.append(left_derivative)

            # Threshold calculation
            # sign_der = np.sign(derivative).astype("int")
            # pos_derivative = np.asarray(derivative)[np.asarray(derivative)>0]
            abs_derivative = np.abs(derivative)
            threshold = np.percentile(abs_derivative, percentile_value)

        # Complete derivative after percetile threshold
        new_derivative = get_index_positions_by_condition(
            abs_derivative, lambda x: x > threshold)
        # The indexes of the derivative list whose values are above the threshold are the indexes corresponding to the pre_RTP
        # elements which are true RTPs. This consitutes an acceptance step

        return new_derivative

    # This is the MAIN algorithm of the function RTP

    # Getting preliminary RTPs, Test sequence and Level Sequence
    if level_signal == 'True':
        pre_RTP, TS = preliminary_RTP(signal, N)
    else:
        TS = preliminary_RTP(signal, N)

    # Getting the indexes of true RTPs
    # si potrebbe far ritornare true_RTP direttamente a questa funzione (?) invece che prima gli indici e fuori true_RTP
    indexes_true_RTP = get_indexes_true_RTP(TS, percentile, n)

    # Creating the true_RTP list
    # true_RTP = [pre_RTP[i] for i in indexes_true_RTP]
    if level_signal == 'True':
        true_RTP = np.intersect1d(pre_RTP, np.asarray(indexes_true_RTP)).tolist()
        return true_RTP
    else:
        return indexes_true_RTP


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def signal_bandpass_filter(data, band_values, fs, order=5):
    lowcut = band_values[0]
    highcut = band_values[1]
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return np.asarray(y)


def full_RTP(signal, bands, fs, N, percentile, n_der, level_signal='True'):
    """
    This function finds the RTPs sequences in the signal channels.
    Input:
        - signal: DataFrame of signal timeseries in each channel
        - bands: frequency bands of interest
        - fs: sampling frequancy
        - N: window length for signal filtering
        - percentile: threshold for selecting true events
    Output:
        - ch_RTP: pd.DataFrame with columns ['Channel'], ['Band'], ['Events']
                  If the analysis is on entire signal: 'Band' = 'all'
                  'Events' is a list of event occurrence times
    """
    ch_RTP = {"Channel": [], "Band": [], "Events": []}
    # Computing the RTP sequence for each channel
    start_time = tm.time()
    progress = 0.
    channels_left = len(signal.columns)
    for channel in signal.columns:
        
        channels_left -= 1                        
        display_progress(start_time, progress, len(signal.columns)-channels_left, len(signal.columns))
        
        # Subtracting the mean signal to neglect Fourier Transform first component
        signal[channel] -= np.mean(signal[channel])
        # Selecting a band
        for i in range(0, len(bands), 1):
            values = bands[i]
            # Checking if the filtering is required
            if ((values[0] < 0) and (values[1] < 0)):
                filtered_data = signal[channel]
                band = 'all'
            # Signal filtering and band indexing the band
            else:
                filtered_data = signal_bandpass_filter(
                    signal[channel], values, fs)
                band = str(values)
            # Creating RTPs sequence
            print(f'Channel {channel}, band {band}')
            if level_signal == 'True':
                RTP_sequence = RTP(signal=filtered_data, sfreq=fs, N=N, percentile=percentile, n=n_der, level_signal=level_signal)
            else:
                RTP_sequence = RTP(signal=filtered_data, sfreq=fs, N=N, percentile=percentile, n=n_der, level_signal=level_signal)
            ch_RTP["Events"].append(RTP_sequence)
            ch_RTP["Band"].append(band)
            ch_RTP["Channel"].append(channel)
    if level_signal == 'True':
        return pd.DataFrame.from_dict(ch_RTP)
    else:
        return pd.DataFrame.from_dict(ch_RTP)


def RTP_extraction(data_sim_list_string, RTP_input_parameters, level_signal):
    
    # Open text file (.txt) with the parameters for RTPs events extraction
    par = []
    with open(RTP_input_parameters, 'r') as file:
        lines = [line.rstrip() for line in file if line.rstrip()]
        for line in lines:
            if not line.startswith('#'):
                par.append(line)

    # Sampling Frequency of the signal
    arr = [float(i.strip()) for i in par[0][1:-1].split(",")]
    fs = np.array(arr)[0]
    # Frequency Bands
    arr1 = [float(i.strip()) for i in par[1][1:-1].split(",")]
    arr2 = [float(i.strip()) for i in par[2][1:-1].split(",")]
    bands = np.stack((arr1, arr2), axis=-1)

    for b in bands:
        if (b[0] < 0 and b[1] > 0) or (b[0] > 0 and b[1] < 0):
            sys.exit('Error: Lower and Upper bound frquency bands must have the same sign')
        elif b[1] < b[0]:
            sys.exit('Error: Upper bound frequency must be greater than lower bound frequency')

    # Window lenght for LS smoothing
    arr = [int(i.strip()) for i in par[3][1:-1].split(",")]
    windows = np.array(arr)

    for w in windows:
        if w <= 0 or type(w) != int:
            sys.exit('Error: Window lenght for LS smoothing must be a non-zero positive integer number')

    # Percentile values
    arr = [int(i.strip()) for i in par[4][1:-1].split(",")]
    percentiles = np.array(arr)

    for p in percentiles:
        if p <= 0 or p >= 100:
            sys.exit('Error: Percentile values must be a positive number between 1 and 99')
            
    # Time-steps to estimate derivative values
    arr = [int(i.strip()) for i in par[5][1:-1].split(",")]
    n_der = np.array(arr)

    for n in n_der:
        if n <= 0 or type(n) != int:
            sys.exit('Error: Time-steps to estimate the derivative must be a positive integer value')

    # Clean temporary variables
    del arr
    del arr1
    del arr2
    
    data_sim_list = data_sim_list_string.split('|')
        
    for data_sim in data_sim_list:
            
        # Load data
        data = pd.read_csv(f'{data_sim}')    
    
        tmp=data_sim.split('/')
        path_data='/'.join(tmp[:-1])
        file_name=tmp[len(tmp)-1].split('.')[0]

        # Save .txt file in the results directory with the parameters used for the event detection
        txt1 = ""

        path = f'{path_data}/{file_name}_Event_Detection'
        name_list = []

        if level_signal == 'True':
            
            for n in n_der:
                
                for window in windows:

                    for percentile_value in percentiles:

                        print(f'Begin single-channel RTP_extraction (windows_length = {window}, percentile = {percentile_value}, n_der = {n})')

                        data = pd.read_csv(f"{path_data}/{file_name}.csv")

                        res = full_RTP(signal=data, bands=bands, fs=fs, N=window, percentile=percentile_value, n_der=n, level_signal=level_signal)

                        # Save Results
                        result_number = 1
                        while os.path.exists(f"{path}/event_detection_window_{window}_percentile_{percentile_value}_{result_number}.csv") == True:
                            result_number += 1

                        # Reading data from file
                        with open(RTP_input_parameters) as fp:
                            txt1 = fp.read()

                        with open(f'{path_data}/{file_name}_Event_Detection/event_detection_parameters_{result_number}.txt', 'w') as fp:
                            fp.write(txt1)

                        res.to_csv(f"{path}/event_detection_window_{window}_percentile_{percentile_value}_{result_number}.csv", index=False)

                        # Create path for save data
                        if os.path.isdir(f'{path}/images') == False:
                            directory = 'images'
                            parent_dir = f'{path}'
                            path_temp = os.path.join(parent_dir, directory)
                            os.makedirs(path_temp, exist_ok=True)

                        # Raster Plot
                        bands_tmp = pd.unique(res["Band"])
                        for band in bands_tmp:
                            events_band = res[res["Band"] == band].reset_index(drop=True)
                            fig, ax = plt.subplots()
                            for c in events_band["Channel"]:
                                ax.eventplot(list(map(int, events_band["Events"][int(c)])), lineoffsets=int(events_band["Channel"][int(c)]), linewidths=0.5, orientation='horizontal', colors='b')
                            new_list_y = range(math.floor(min(list(map(float, events_band["Channel"])))), math.ceil(max(list(map(float, events_band["Channel"]))))+1)
                            ax.set_yticks(new_list_y)
                            ax.set_ylabel('Channels')
                            ax.set_xlabel('Time')
                            ax.set_title(f'Raster Plot for band {band} (window = {window}, percentile = {percentile_value})')
                            
                            img = fig2img(fig)
                            
                            img.save(f'{path}/images/event_detection_band_{band}_window_{window}_percentile_{percentile_value}_{result_number}_raster_plot.png', quality = 85)
                            # plt.savefig(f'{path}/images/event_detection_band_{band}_window_{window}_percentile_{percentile_value}_{result_number}_raster_plot.png')
                            img.close()

                        name_list.append(f'event_detection_window_{window}_percentile_{percentile_value}_{result_number}')

                        print(f'Finish single-channel RTP_extraction (windows_length = {window}, percentile = {percentile_value}, n_der = {n})')

            return name_list

        else:
            
            for n in n_der:

                for percentile_value in percentiles:

                    print(f'Begin single-channel RTP_extraction (percentile = {percentile_value}, n_der = {n})')

                    data = pd.read_csv(f"{path_data}/{file_name}.csv")

                    res = full_RTP(signal=data, bands=bands, fs=fs, N='None', percentile=percentile_value, n_der=n, level_signal=level_signal)

                    # Save Results
                    result_number = 1
                    while os.path.exists(f"{path}/event_detection_percentile_{percentile_value}_{result_number}.csv") == True:
                        result_number += 1

                    # Reading data from file
                    with open(RTP_input_parameters) as fp:
                        txt1 = fp.read()

                    with open(f'{path_data}/{file_name}_Event_Detection/event_detection_parameters_{result_number}.txt', 'w') as fp:
                        fp.write(txt1)

                    res.to_csv(f"{path}/event_detection_percentile_{percentile_value}_{result_number}.csv", index=False)
                    
                    # Create path for save data
                    if os.path.isdir(f'{path}/images') == False:
                        directory = 'images'
                        parent_dir = f'{path}'
                        path_temp = os.path.join(parent_dir, directory)
                        os.makedirs(path_temp, exist_ok=True)

                    # Raster Plot
                    bands_tmp = pd.unique(res["Band"])
                    for band in bands_tmp:
                        events_band = res[res["Band"] == band].reset_index(drop=True)
                        fig, ax = plt.subplots()
                        for c in events_band["Channel"]:
                            ax.eventplot(list(map(int, events_band["Events"][int(c)])), lineoffsets=int(events_band["Channel"][int(c)]), linewidths=0.5, orientation='horizontal', colors='b')
                        new_list_y = range(math.floor(min(list(map(float, events_band["Channel"])))), math.ceil(max(list(map(float, events_band["Channel"]))))+1)
                        ax.set_yticks(new_list_y)
                        ax.set_ylabel('Channels')
                        ax.set_xlabel('Time')
                        ax.set_title(f'Raster Plot for band {band} (percentile = {percentile_value})')
                        
                        img = fig2img(fig)
                            
                        img.save(f'{path}/images/event_detection_band_{band}_percentile_{percentile_value}_{result_number}_raster_plot.png', quality = 85)
                        # plt.savefig(f'{path}/images/event_detection_band_{band}_percentile_{percentile_value}_{result_number}_raster_plot.png')
                        img.close()
                        
                    name_list.append(f'event_detection_percentile_{percentile_value}_{result_number}')

                    print(f'Finish single-channel RTP_extraction (percentile = {percentile_value}, n_der = {n})')

            return name_list 
    
