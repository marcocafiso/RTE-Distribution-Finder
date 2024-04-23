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


###################################################################################################################
############################################## EVENT DETECTION ####################################################
###################################################################################################################

def RTP(signal,sfreq,N= None, percentile = 90, level_signal = 'True', pos_der = 'False'):
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

        if N == None:          #se non metto nulla python dà errore
            N = len(TS)        #lo lascio nel caso dovesse servire media su tutta serie temporale   
            
        if level_signal == 'True':
            
            # Smoothing TS with a moving average of lenght N
            LS = ndif.uniform_filter1d(TS, N, mode='wrap', origin=0) # considero come se ci fossero condizioni periodiche al bordo
            
            # Finding intersections between signals as the indexes corrisponding to sign changes between two consecutive elements
            # in the array TS - LS
            if np.isnan(TS).any() == True:
                pre_RTP = []
            else:
                pre_RTP = np.argwhere(np.diff(np.sign(TS - LS))).flatten()
            
            return pre_RTP,TS,LS
        
        else:
            return TS
            

 
    def get_indexes_true_RTP(TS, percentile_value):
        """
        This function looks for statistically significant RTPs (step 2).
        
        From the signal (TS - LS or only TS) we estimated is derivative for each
        time-step by using the previous and next 12 points. 
        After we extract from the distribution of the derivative only the time-steps
        regarding the desired percentile value of the distribution.
        
        """
        S = TS
        # S = TS - LS
        n = 12               #mi sembra ne prenda 12 di punti successivi, non 5
        derivative = []
        step_size = 1/sfreq  #frequenza di campionamento per calcolare step
        
        # Now i have to calculate up to the fifth value around a pre_RTP. Thus now I calculate this derivative on the S 
        # obtained above (calcola derivata facendo la media delle pendenze tra punti sequenzialmente e simmetricamente più lontani da pre_RTP(fino a 12) )
        if len(S) > 0:
            
            # Derivative for extreme left element
            right_derivative = np.divide(S[1]-S[0],step_size)
            derivative.append(right_derivative)
            
            for el in range(1,n):
                sum_array = [np.divide(S[el+i]-S[el-i],(2*i)*step_size) for i in range(1, el+1)]
                derivative.append(1/len(sum_array) * np.sum(sum_array))  
            
            for el in range(n,len(S)-n):
                sum_array = [np.divide(S[el+i]-S[el-i],(2*i)*step_size) for i in range(1,n+1)]
                derivative.append(1/len(sum_array) * np.sum(sum_array)) 
                    
            for el in range(len(S)-n,len(S)-1):
                sum_array = [np.divide(S[el+i]-S[el-i],(2*i)*step_size) for i in range(1,len(S)-el)]
                derivative.append(1/len(sum_array) * np.sum(sum_array))                                      
            
            # Derivative for extreme right element
            left_derivative = np.divide(S[len(S)-1]-S[len(S)-2],step_size)
            derivative.append(left_derivative)
            
            #Threshold calculation
            #sign_der = np.sign(derivative).astype("int")  
            # pos_derivative = np.asarray(derivative)[np.asarray(derivative)>0]               
            abs_derivative = np.abs(derivative)
            threshold = np.percentile(abs_derivative,percentile_value)
        
        # Complete derivative after percetile threshold
        new_derivative = get_index_positions_by_condition(abs_derivative,lambda x: x > threshold)
        
        # Only positive values of the derivative after percentile threshold
        new_pos_derivative = []
        for i in new_derivative:
            
            if derivative[i] > 0:
                
                new_pos_derivative.append(i)
            
        
        #exit(0) # se lascio exit(0) mi uccide il kernel
        
        # The indexes of the derivative list whose values are above the threshold are the indexes corresponding to the pre_RTP
        #elements which are true RTPs. This consitutes an acceptance step
        
        return new_derivative, new_pos_derivative, derivative
   

    ##### This is the MAIN algorithm of the function RTP
    
    # Getting preliminary RTPs, Test sequence and Level Sequence
    if level_signal == 'True':
        pre_RTP, TS, LS = preliminary_RTP(signal,N)
    else:
        TS = preliminary_RTP(signal,N)
    
    # Getting the indexes of true RTPs
    indexes_true_RTP, indexes_true_RTP_pos, derivative = get_indexes_true_RTP(TS,percentile) #si potrebbe far ritornare true_RTP direttamente a questa funzione (?) invece che prima gli indici e fuori true_RTP
    
    # Creating the true_RTP list 
    # true_RTP = [pre_RTP[i] for i in indexes_true_RTP]
    if level_signal == 'True':
        if pos_der == 'False':
            true_RTP = np.intersect1d(pre_RTP, np.asarray(indexes_true_RTP)).tolist()
        else:
            true_RTP = np.intersect1d(pre_RTP, np.asarray(indexes_true_RTP_pos)).tolist()
        #true_RTP_sign_derivative = [sign_derivative[i] for i in indexes_true_RTP]
        return true_RTP, TS, LS, derivative
    else:
        if pos_der == 'False':
            return indexes_true_RTP, TS, derivative
        else:
            return indexes_true_RTP_pos, TS, derivative



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



def full_RTP(signal,bands,fs,N,percentile, level_signal = 'True', pos_der = 'False'):
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
    ch_RTP = {"Channel":[],"Band":[],"Events" : []}
    # Computing the RTP sequence for each channel 
    for channel in signal.columns:
        # Subtracting the mean signal to neglect Fourier Transform first component 
        signal[channel] -= np.mean(signal[channel])
        # Selecting a band
        for i in range(0,len(bands),1):
            values = bands[i]
            # Checking if the filtering is required
            if ((values[0]<0) and (values[1]<0)):
                filtered_data=signal[channel]
                band = 'all'
            # Signal filtering and band indexing the band
            else:
                filtered_data = signal_bandpass_filter(signal[channel], values, fs)  
                band = str(values)
            # Creating RTPs sequence
            print(f'Channel {channel}, band {band}')
            if level_signal == 'True':
                RTP_sequence, TS, LS, derivative = RTP(signal = filtered_data, sfreq = fs, N=N, percentile = percentile, level_signal=level_signal, pos_der=pos_der)
            else:
                RTP_sequence, TS, derivative = RTP(signal = filtered_data, sfreq = fs, N=N, percentile = percentile, level_signal=level_signal, pos_der=pos_der)
            ch_RTP["Events"].append(RTP_sequence)
            ch_RTP["Band"].append(band)
            ch_RTP["Channel"].append(channel)
    if level_signal == 'True':
        return pd.DataFrame.from_dict(ch_RTP), TS, LS, derivative
    else:
        return pd.DataFrame.from_dict(ch_RTP), TS, derivative



def RTP_extraction(data_sim, RTP_input_parameters, level_signal = 'True', pos_der = 'False'):
    
    tmp=data_sim.split('/')
    path_data='/'.join(tmp[:-1])
    file_name=tmp[len(tmp)-1]
    
    # Open text file (.txt) with the parameters for RTPs events extraction
    par = []
    with open(RTP_input_parameters, 'r') as file:
        lines = [line.rstrip() for line in file if line.rstrip()]
        for line in lines:
            if not line.startswith('#'):
                par.append(line)
                

    # Sampling Frequency of the signal
    arr = [float(i.strip()) for i in par[0][1:-1].split(",")]
    fs=np.array(arr)[0]
    # Frequency Bands
    arr1 = [float(i.strip()) for i in par[1][1:-1].split(",")]
    arr2 = [float(i.strip()) for i in par[2][1:-1].split(",")]
    bands=np.stack((arr1, arr2), axis=-1)
    
    for b in bands:
        if (b[0] < 0 and b[1] > 0) or (b[0] > 0 and b[1] < 0):
            sys.exit('Error: Lower and Upper bound frquency bands must have the same sign')
        elif b[1] < b[0]:
            sys.exit('Error: Upper bound frequency must be greater than lower bound frequency')
    
    
    # Window lenght for LS smoothing 
    arr = [int(i.strip()) for i in par[3][1:-1].split(",")]
    windows=np.array(arr)
    
    for w in windows:
        if w <= 0:
            sys.exit('Error: Window lenght for LS smoothing must be a non-zero positive integer number')
    
    # Percentile values
    arr = [int(i.strip()) for i in par[4][1:-1].split(",")]
    percentiles=np.array(arr)
    
    for p in percentiles:
        if p <= 0 or p >= 100:
            sys.exit('Error: Percentile values must be a positive number between 1 and 99')
    
    # Clean temporary variables
    del arr 
    del arr1 
    del arr2
    
    # Save .txt file in the results directory with the parameters used for the event detection    
    txt1 = ""
 
    path = f'{path_data}/{file_name}_Event_Detection'
    name_list = []
    TS_name_list = []  
    der_name_list = []          
    if level_signal == 'True':
        LS_name_list = []
        for window in windows:

            for percentile_value in percentiles:
                
                print(f'Begin single-channel RTP_extraction (windows_length = {window}, percentile = {percentile_value})')
                
                data = pd.read_csv(f"{path_data}/{file_name}.csv")               
 
                res, TS, LS, derivative = full_RTP(signal = data,bands = bands, fs = fs, N = window,percentile = percentile_value, level_signal=level_signal, pos_der=pos_der)
                
                # Save Results
                result_number = 1                
                while os.path.exists(f"{path}/event_detection_window_{window}_percentile_{percentile_value}_{result_number}.csv") == True:
                    result_number += 1
                
                # Reading data from file
                with open(RTP_input_parameters) as fp:
                    txt1 = fp.read()
                    
                with open (f'{path_data}/{file_name}_Event_Detection/event_detection_parameters_{result_number}.txt', 'w') as fp:
                    fp.write(txt1)
                
                res.to_csv(f"{path}/event_detection_window_{window}_percentile_{percentile_value}_{result_number}.csv", index=False)
                
                np.savetxt(f"{path}/test_signal_window_{window}_percentile_{percentile_value}_{result_number}.txt", TS)
                
                np.savetxt(f"{path}/level_signal_window_{window}_percentile_{percentile_value}_{result_number}.txt", LS)                    
                
                np.savetxt(f"{path}/derivative_window_{window}_percentile_{percentile_value}_{result_number}.txt", derivative)           
                
                
                # Create path for save data
                if os.path.isdir(f'{path}/images') == False:
                    directory = 'images'
                    parent_dir = f'{path}'
                    path_temp = os.path.join(parent_dir, directory) 
                    os.makedirs(path_temp, exist_ok= True)
                
                # Raster Plot
                bands_tmp = pd.unique(res["Band"])
                for band in bands_tmp:
                    events_band = res[res["Band"] == band].reset_index(drop=True)
                    plt.figure()
                    for c in events_band["Channel"]:
                        plt.eventplot(list(map(int,events_band["Events"][int(c)])), lineoffsets=int(events_band["Channel"][int(c)]), linewidths = 0.5, orientation='horizontal', colors='b')
                    new_list_y = range(math.floor(min(list(map(float,events_band["Channel"])))), math.ceil(max(list(map(float,events_band["Channel"]))))+1)
                    plt.yticks(new_list_y)
                    plt.ylabel('Channels')
                    plt.xlabel('Time')
                    plt.title(f'Raster Plot for band {band} (window = {window}, percentile = {percentile_value})')
                    plt.savefig(f'{path}/images/event_detection_band_{band}_window_{window}_percentile_{percentile_value}_{result_number}_raster_plot.png', dpi=600)
                    plt.show()
                    
                name_list.append(f'event_detection_window_{window}_percentile_{percentile_value}_{result_number}')
                
                TS_name_list.append(f'test_signal_window_{window}_percentile_{percentile_value}_{result_number}')
                
                LS_name_list.append(f'level_signal_window_{window}_percentile_{percentile_value}_{result_number}')                    
                
                der_name_list.append(f'derivative_window_{window}_percentile_{percentile_value}_{result_number}')
                
                print(f'Finish single-channel RTP_extraction (windows_length = {window}, percentile = {percentile_value})')
                
        return name_list, TS_name_list, LS_name_list, der_name_list
    
    else:
        
        for percentile_value in percentiles:
            
            print(f'Begin single-channel RTP_extraction (percentile = {percentile_value})')
            
            data = pd.read_csv(f"{path_data}/{file_name}.csv")               

            res, TS, derivative = full_RTP(signal = data,bands = bands, fs = fs, N = 'None', percentile = percentile_value, level_signal=level_signal, pos_der=pos_der)
            
            # Save Results
            result_number = 1                
            while os.path.exists(f"{path}/event_detection_percentile_{percentile_value}_{result_number}.csv") == True:
                result_number += 1
            
            # Reading data from file
            with open(RTP_input_parameters) as fp:
                txt1 = fp.read()
                
            with open (f'{path_data}/{file_name}_Event_Detection/event_detection_parameters_{result_number}.txt', 'w') as fp:
                fp.write(txt1)
            
            res.to_csv(f"{path}/event_detection_percentile_{percentile_value}_{result_number}.csv", index=False)
            
            np.savetxt(f"{path}/test_signal_percentile_{percentile_value}_{result_number}.txt", TS)                   
            
            np.savetxt(f"{path}/derivative_percentile_{percentile_value}_{result_number}.txt", derivative)           
            
            
            # Create path for save data
            if os.path.isdir(f'{path}/images') == False:
                directory = 'images'
                parent_dir = f'{path}'
                path_temp = os.path.join(parent_dir, directory) 
                os.makedirs(path_temp, exist_ok= True)
            
            # Raster Plot
            bands_tmp = pd.unique(res["Band"])
            for band in bands_tmp:
                events_band = res[res["Band"] == band].reset_index(drop=True)
                plt.figure()
                for c in events_band["Channel"]:
                    plt.eventplot(list(map(int,events_band["Events"][int(c)])), lineoffsets=int(events_band["Channel"][int(c)]), linewidths = 0.5, orientation='horizontal', colors='b')
                new_list_y = range(math.floor(min(list(map(float,events_band["Channel"])))), math.ceil(max(list(map(float,events_band["Channel"]))))+1)
                plt.yticks(new_list_y)
                plt.ylabel('Channels')
                plt.xlabel('Time')
                plt.title(f'Raster Plot for band {band} (percentile = {percentile_value})')
                plt.savefig(f'{path}/images/event_detection_band_{band}_percentile_{percentile_value}_{result_number}_raster_plot.png', dpi=600)
                plt.show()
                
            name_list.append(f'event_detection_percentile_{percentile_value}_{result_number}')
            
            TS_name_list.append(f'test_signal_percentile_{percentile_value}_{result_number}')                   
            
            der_name_list.append(f'derivative_percentile_{percentile_value}_{result_number}')
            
            print(f'Finish single-channel RTP_extraction (percentile = {percentile_value})')
            
        return name_list, TS_name_list, der_name_list
    
    
    

###################################################################################################################
############################################ AVALANCHES AND WTs ###################################################
###################################################################################################################

def ordering_time_events(events_band):
    """
    This function is used to create a new df cointaining all the events sorted in time for a selected band.
    This allows an easier implementation of the new_multichannel_RTP function to detect global events.
    """    
    df= events_band.copy()
    df=df.reset_index()
    header = ['Channels', 'Time_Events']
    channels = []
    events = []
    new_df_list = []
    for j in range(df.shape[0]):        
        events_row = df['Events'].iloc[j]        
        for event in events_row:            
            events.append(event)
            channels.append(df['Channel'].iloc[j])

    new_df_list.append(channels)
    new_df_list.append(events)

    new_df = pd.DataFrame(new_df_list).transpose()
    new_df.columns = header

    new_df.sort_values(by = ['Time_Events'], inplace=True)

    df_new=new_df.drop(0).reset_index(drop=True)
    df_new['Time_Events'] = df_new['Time_Events'].astype(int)
    
    return df_new


def avalanche_definition(RTP_seq, dt_av, subsequence, subsequence_start):
    '''
    This function creates an array containing the number of events in each bin of size dt_av
    (dt_av=integer that defines the time interval where events are consider simultaneous) for each band.
    For each time interval the information about the channels involved is saved too.
    Input:
        - RTP_seq: DataFrame containing the sequence of events
        - dt_av: time interval to be used to define the avalanche
    Output:
        - avalanche: DataFrame with columns: ['Band'], ['Counts'] and ['Channels involved']
    '''
    bands = pd.unique(RTP_seq["Band"])
    avalanche = {"Band":[], "Counts": [], "Channels involved": []}
    for band in bands: 
        # Select only the events of the desired band
        events_band = RTP_seq[RTP_seq["Band"] == band]
        # Creating a new DF with sorted time events  
        ordered_df = ordering_time_events(events_band)
        if (type(ordered_df)==type(None)):
        	continue
        ordered_df=ordered_df.reset_index()
        # max_t=0
        # for i in range(len(RTP_seq["Events"].values)):
        #     if(((len(RTP_seq["Events"].values[i])>0) and (max(RTP_seq["Events"].values[i])>max_t))):
        #         max_t=max(RTP_seq["Events"].values[i])
        
        max_t = ordered_df['Time_Events'].values[-1]  
        time=np.arange(max_t+1)
        
        # Creating the list of time intervals 
        time_intervals=[]
        
        # Modifica Marco
        if subsequence == 'full':
            
            for i in range(subsequence_start,len(time), dt_av):
                time_intervals.append(time[i:i+dt_av].tolist())
            count=np.zeros(len(time_intervals))
            channels=[[] for i in range(len(time_intervals))]

            # Counting how many events fall into each time-interval and finding the associated channel
            for index, i in enumerate(time_intervals):
                for j in i:
                    condition = ordered_df["Time_Events"].values==j
                    count[index]+=(condition).sum()
                    for k in np.argwhere(condition):
                        channels[index].append(ordered_df["Channels"][k].values[0])
        else:
            
            for i in range(subsequence_start[subsequence-1],len(time), dt_av):
                time_intervals.append(time[i:i+dt_av].tolist())
            count=np.zeros(len(time_intervals))
            channels=[[] for i in range(len(time_intervals))]

            # Counting how many events fall into each time-interval and finding the associated channel
            for index, i in enumerate(time_intervals):
                for j in i:
                    condition = ordered_df["Time_Events"].values==j
                    count[index]+=(condition).sum()
                    for k in np.argwhere(condition):
                        channels[index].append(ordered_df["Channels"][k].values[0])


        avalanche["Counts"].append(count.astype("int"))
        avalanche["Band"].append(band)
        avalanche["Channels involved"].append(channels)
        
    avalanche = pd.DataFrame.from_dict(avalanche)
    
    # print(avalanche)    

    return avalanche    
       
        

def avalanche_identification_and_size(counts, N_av, dt_av):
    '''
    This function implements an the algorithm for avalanche identification and computes its size.
    See Ribeiro et al. PLoS ONE 5(11): e14129 for details.
    Input:
        - sequence of number of events in each time interval of lenght dt_av
        - N_av: minimum number of events to define an avalanche 
        - dt_av: 
    Output: 
        a dictionary contening
        - tau_pp: time duration between two consecutive avalanches
        - tau_pn: avalanche time duration 
        - tau_np: no avalanche time duration
        - size: total number of events for each avalanche
        a dictionary contening
        - t_p: starting times of the avalanches
        - t_pn: cumsum of tau_pn
        - t_np: cumsum of tau_np
        - t_ae: all events ({t_p}U{t_n})
    '''
    
    av_dict={"tau_pp":[], "tau_pn":[], "tau_np":[], "size":[]}
    t_dfa = {"t_p":[], "t_pn":[], "t_np":[], "t_ae":[]}
    # Here we extend counts array adding 0 in the initial and final position when not the if_condition is satisfied.
    # This prevents from erroneus identification of tau_p and tau_n times in the following For Loop.
    if ((counts[0]!=0) or (counts[-1]!=0)):
        counts.insert(0,0)
        counts.insert(len(counts),0)
    # t_p/t_n represents the starting/finishing time of the avalanche.
    t_p = []
    t_n = []
    # Here the condition for identifying tau_p and tau_n are defined.
    for i in range(0, len(counts)-1):
        condition_up = (counts[i]<N_av) & (counts[i+1]>N_av-1)
        if (condition_up):
            t_p.append((i+1)*dt_av)        
        condition_down = (counts[i]>N_av-1) & (counts[i+1]<N_av)
        if(condition_down):
            t_n.append((i+1)*dt_av)
    # Calculation of time duration variables that characterize the avalanche    
    tau_pp=np.diff(np.asarray(t_p))
    tau_pn=np.asarray(t_n)-np.asarray(t_p)
    tau_np=np.zeros(len(tau_pp-1)).astype("int")
    for i in range(len(t_p)-1):
        tau_np[i]= t_p[i+1] - t_n[i]
    # Calculation of avalanche size
    size=np.zeros(len(t_p)).astype("int")   
    for index, (i, j) in enumerate(zip(t_p, t_n)):
        size[index]+=sum(counts[i:j])
    
    # Building the final dictionary and DF 
    av_dict["tau_pp"].append(list(tau_pp))   #devo mettere list perchè senno quando vado a salvare DF in CSV mi toglie le       
    av_dict["tau_pn"].append(list(tau_pn))   #virgole, inotre ricorda che per DF letto da CSV le colonne vanno trasformate
    av_dict["tau_np"].append(list(tau_np))   # da stringhe a valori numerici tramite literal_eval()
    av_dict["size"].append(list(size))
    AV=pd.DataFrame.from_dict(av_dict)
    
    t_dfa["t_p"].append(list(t_p))                  #devo mettere list perchè senno quando vado a salvare DF in CSV mi toglie le       
    t_dfa["t_pn"].append(list(np.cumsum(tau_pn)))   #virgole, inotre ricorda che per DF letto da CSV le colonne vanno trasformate
    t_dfa["t_np"].append(list(np.cumsum(tau_np)))   # da stringhe a valori numerici tramite literal_eval()
    t_dfa["t_ae"].append(list(sorted(t_p + t_n)))
    T_DFA=pd.DataFrame.from_dict(t_dfa, dtype=int)
    
    return T_DFA, AV



def Waiting_Times(RTP_seq, dt_av, N_av, subsequence, subsequence_start, path_data, file_name, pipeline):
    """
    This function returns WT DataFrame for each band and bin size which defines the dt_av in avalanche_definition() function.
    """
    avalanche=avalanche_definition(RTP_seq, dt_av, subsequence, subsequence_start)
    bands = pd.unique(avalanche["Band"])
    band_av=[]
    for index, band in enumerate(bands):
        avalanche_band= avalanche[avalanche["Band"] == band] 
        T_DFA, AV=avalanche_identification_and_size(avalanche_band["Counts"][index].tolist(), N_av, dt_av)
        
        # Create path for save data
        if os.path.isdir(f'{path_data}/WT') == False:
            directory = 'WT'
            parent_dir = f'{path_data}'
            path = os.path.join(parent_dir, directory) 
            os.makedirs(path, exist_ok= True)        
        
        # Save data
        if (pipeline[7] == 'y' and pipeline[8] == 'n') or (pipeline[7] == 'y' and pipeline[8] != 'n' and pipeline[10] != 'wt'):
            AV.to_csv(f"{path_data}/WT/WT_subsequence_{subsequence}_dt_av_{dt_av}_N_av_{N_av}_band_{band}.csv",  index=False) #index=False removes "Unnamed" cols when creating new DFs
        elif (pipeline[7] == 'n' and pipeline[8] != 'n' and pipeline[10] == 'wt'):
            T_DFA.to_csv(f'{path_data}/WT/t_subsequence_{subsequence}_dt_av_{dt_av}_N_av_{N_av}_band_{band}.csv',index=False)
        else:
            AV.to_csv(f"{path_data}/WT/WT_subsequence_{subsequence}_dt_av_{dt_av}_N_av_{N_av}_band_{band}.csv",  index=False) #index=False removes "Unnamed" cols when creating new DFs
            T_DFA.to_csv(f'{path_data}/WT/t_subsequence_{subsequence}_dt_av_{dt_av}_N_av_{N_av}_band_{band}.csv',index=False)

        band_av.append(band)
        
    avalanche.to_csv(f"{path_data}/WT/avalanche_structure_subsequence_{subsequence}.csv",  index=False)
    return band_av




###################################################################################################################    
############################################ DFA & DE ANALYSIS ####################################################
###################################################################################################################

# CTRW
def CTRW(sim_len,event_times,jump_type = str):
    """
    This function builds the tracjetories for the implementation of the EDDiS method.
    Input:
        - sim_len: total simulation length  
        - event_times: list of times for single channel/global events from RTP_seq/MC_RTPs DataFrame
        - jump_type: 'SJ', 'AJ', 'SV'. Stands for symmetric jump, asymmetric jump and symmetric velocity
        
    Output:
        - X: final trajectory
    """
    T = np.linspace(0, sim_len +1, sim_len+2, dtype=int)
    X = np.zeros(len(T), dtype=int)
    #print('RTP_list', RTPs_list)
    
    # Here we call an array filled with 0s or 1s according to a coin toss probability
    # we set the seed in order to have the same array  
    choice_list = np.random.randint(low = 0, high = 2, size = len(event_times)+1)
    if jump_type == 'SJ':
        
        for i,element in enumerate(event_times):
            
            if choice_list[i] == 0: 
                X[element] = 1
            else:
                X[element] = -1

    elif jump_type == 'SV':
        for i, element in enumerate(event_times):
            if i == 0:
                if choice_list[i] == 0: 
                    X[:element] = 1
                else:
                    X[:element] = -1
            elif i == len(event_times)-1: #devo fare doppio controllo per sistemare anche sequenza dopo ultimo evento
                if choice_list[i] == 0: 
                    X[event_times[i-1]:element] = 1
                else:
                    X[event_times[i-1]:element] = -1
                    
                if choice_list[-1] == 0: 
                    X[element:] = 1
                else:
                    X[element:] = -1
            else:
                if choice_list[i] == 0: 
                    X[event_times[i-1]:element] = 1
                else:
                    X[event_times[i-1]:element] = -1
            
    else:
        X[event_times] = 1
            
    return X



############################################## DFA Analysis #######################################################

# giocare con X.size//..., numero di punti fit e slice dei dati su cui fare fit per ottenere risultati migliori
# dipende da numero dati X
def DFA_channel_band(X, path_data, label, H_exp):
    """
    This function applies the MFDFA routine to obtain the DFA and computes the Hurst coefficient H.
    Input:
        - X: trajectory returned from CTRW()
    Output:
        - 
    
    """
    # Create an array of logarithmic spaced values
    lag = np.unique(np.logspace(1, np.log10(X.size // 10), 50).astype(int)+1)
    # Apply DFA routine to the trajectory X
    lag, dfa = MFDFA(X, lag = lag, q = 2, order = 1)
    # Linear fit in logarithmic scale to find H
    # popt = np.polyfit(np.log10(lag), np.log10(dfa),1)
    plf = pwlf.PiecewiseLinFit(np.log10(lag), np.log10(np.concatenate(dfa)), seed=10)   
    breaks = plf.fit(3)
    
    r_2 = plf.r_squared()
       
    if r_2 > 0.99:
        breaks = plf.fit(2)
        r_2 = plf.r_squared()
        
    # print(f'R_2 = {r_2}')
    
    beta = plf.beta
    
    p_values = plf.p_values()

    H = plf.calc_slopes()
        
    #PLOT
    print(f'H = {H}')
    
    xdata=np.linspace(min(lag), max(lag), 10000)   
    
    y_hat = plf.predict(np.log10(xdata))  

    def f(x, exp, a):
        return a*x**exp

    plt.figure(figsize=(8,7))
    plt.xscale('log')
    plt.yscale('log')
    # plt.plot(xdata, f(xdata, *popt), color='red')
    plt.plot(xdata, 10**y_hat, color='red')
    plt.scatter(lag, dfa)
    plt.plot(xdata, f(xdata, H_exp, dfa[-1]/(xdata[-1]**H_exp)), color='green')
    plt.title(label+f': H = {np.around(H,3)}')
    plt.xlabel(r'${\Delta t}$')
    plt.ylabel('DFA')
    save_name=label.replace(" ", "_")
    #print(type(dfa), type(lag))
    #print(np.shape(dfa), np.shape(lag))
    
    # Create path for save images
    if os.path.isdir(f'{path_data}/DFA/images') == False:
        directory = "images"
        parent_dir = f'{path_data}/DFA'
        path = os.path.join(parent_dir, directory) 
        os.makedirs(path, exist_ok= True) 
    
    
    # Save plots
    save_data_plot=np.array([lag, dfa.reshape(len(dfa))]).T    
    np.savetxt(f'{path_data}/DFA/images/{save_name}.txt', save_data_plot)
    plt.savefig(f'{path_data}/DFA/images/{save_name}.png', dpi=600) 
    plt.show()
    return H, p_values, r_2, beta   #,lag,dfa #<--------------------------------- scegliere cosa ritornare



################################################ DE Analysis ######################################################    

def DE(X, path_data, label):
    """
    This functions implements the diffusion entropy (DE) algorithm.
    Input:
        - X: trajectory returned from CTRW()
    Output:
        - E: entropy values
        - times: array of time lag used to split the time series into overlapping time window
        - delta: PDF self-similarity index
    
    """
    # Computes the diffusive variable
    s = np.cumsum(X)
    E = []
    # hist_bins= []

    # Create an array of logarithmic spaced values
    time_lags = list(np.unique(np.logspace(1, np.log10(X.size // 5), 50).astype(int)+1))
     
    time_lags=time_lags[:-1]

    # For each time lag the PDF and the associated entropy are evalueted  
    for lag in time_lags:
        len_max= len(s)-lag
        Y=np.zeros(len_max)
        for i in range(len_max):
            Y[i]=s[i+lag]-s[i]
        # Y distribution is centered around the origin to better compute E (especially in the case of 'AJ' trajectories) Non serve!
        # Y_mean=np.mean(Y)
        # Y=Y-Y_mean
        # Count the number of particles found in bins     
        # hist_bins.append(int(max(Y)-min(Y)))
        counts,_ = np.histogram(Y, bins = int(max(Y)-min(Y))) #bins deve essere t.c. bin sia frazione varianza distr.
        # Probability 
        D_bin=(max(Y)-min(Y))/int(max(Y)-min(Y))
        prob = counts/(len_max*D_bin)
        mask = prob>0
        prob = prob[mask]
        # Entropy
        E.append(-np.sum(prob*np.log10(prob)*D_bin))
        
    plf = pwlf.PiecewiseLinFit(np.log10(time_lags[:]), E[:], seed=10)   
    breaks = plf.fit(3)

    r_2 = plf.r_squared()

    # if r_2 > 0.99:
    #     breaks = plf.fit(2)
    #     r_2 = plf.r_squared()

    # print(f'R_2 = {r_2}')

    beta = plf.beta

    p_values = plf.p_values()

    delta = plf.calc_slopes()

    #PLOT 
    def f(x, m, q):
        return m*np.log10(x)+q

    plt.figure(figsize=(8,7))
    plt.plot(time_lags,E, '+')
    x=np.linspace(min((time_lags)[:]), max((time_lags)[:]), 10000)
    y_hat = plf.predict(np.log10(x)) 
    print(f'delta = {delta}')
    plt.plot(x, y_hat, color='red')
    plt.xlabel(r'${\Delta t}$')
    plt.title(label + f': delta = {np.around(delta,3)}')
    plt.ylabel('DE')
    plt.xscale('log')
    save_name=label.replace(" ", "_")
    
    # Create path for save images
    if os.path.isdir(f'{path_data}/DE/images') == False:
        directory = 'images'
        parent_dir = f'{path_data}/DE'
        path = os.path.join(parent_dir, directory) 
        os.makedirs(path, exist_ok= True) 
    
    # Save plots       
    save_data_plot=np.array([time_lags, E]).T
    np.savetxt(f'{path_data}/DE/images/{save_name}.txt', save_data_plot)
    plt.savefig(f'{path_data}/DE/images/{save_name}.png', dpi=600) 
    plt.show()
    return delta, p_values, r_2, beta #, E, times<-----scegliere cosa ritonare



################################################ DFA/DE on different data ######################################################         
        

def avalanche_WT_DFA_DE(bands, dt_av, N_av, path_data, file_name, jump, sim_len, subsequence, flag, H_exp):        
    H_single_ch={"Band":[], "tau":[], "H": [], "beta": [], "p-values": [], "R^2": []}
    delta_single_ch={"Band":[],"tau":[], "Delta": [], "beta": [], "p-values": [], "R^2": []}
    for band in bands:
        df=pd.read_csv(f"{path_data}/WT/t_subsequence_{subsequence}_dt_av_{dt_av}_N_av_{N_av}_band_{band}.csv")
        for data in df.columns:
            
            if type(df[data].iat[0]) == str:
                lista = list(map(int,df[data].iat[0][1:-1].split(',')))
            else:
                lista = df[data].iat[0]
            
            X = CTRW(sim_len,lista,jump)
            
            if flag == 'DFA':
                print(f'H for subsequence {subsequence} dt_av = {dt_av}, N_av = {N_av} and band {band} on {data}')
                h, p_vals, r_2, beta = DFA_channel_band(X, path_data, f'subsequence_{subsequence} jump_{jump} dt_av_{dt_av} N_av_{N_av} on {data}', H_exp)

                H_single_ch["Band"].append(band)
                H_single_ch["tau"].append(data)
                H_single_ch["H"].append(h)
                H_single_ch["beta"].append(beta)
                H_single_ch["p-values"].append(p_vals)
                H_single_ch["R^2"].append(r_2)
                
            elif flag == 'DE':
                print(f'delta for subsequence {subsequence} dt_av = {dt_av}, N_av = {N_av} and band {band} on {data}')
                d, p_vals, r_2, beta = DE(X, path_data, f'subsequence_{subsequence} jump_{jump} dt_av_{dt_av} N_av_{N_av} on {data}')
                
                delta_single_ch["Band"].append(band)
                delta_single_ch["tau"].append(data)
                delta_single_ch["Delta"].append(d)
                delta_single_ch["beta"].append(beta)
                delta_single_ch["p-values"].append(p_vals)
                delta_single_ch["R^2"].append(r_2)
                
            else:
                print(f'H for subsequence {subsequence} dt_av = {dt_av}, N_av = {N_av} and band {band} on {data}')
                h, p_vals, r_2, beta = DFA_channel_band(X, path_data, f'subsequence_{subsequence} jump_{jump} dt_av_{dt_av} N_av_{N_av} on {data}', H_exp)

                H_single_ch["Band"].append(band)
                H_single_ch["tau"].append(data)
                H_single_ch["H"].append(h)
                H_single_ch["beta"].append(beta)
                H_single_ch["p-values"].append(p_vals)
                H_single_ch["R^2"].append(r_2)
                
                print(f'delta for subsequence {subsequence} dt_av = {dt_av}, N_av = {N_av} and band {band} on {data}')
                d, p_vals_d, r_2_d, beta_d = DE(X, path_data, f'subsequence_{subsequence} jump_{jump} dt_av_{dt_av} N_av_{N_av} on {data}')
                
                delta_single_ch["Band"].append(band)
                delta_single_ch["tau"].append(data)
                delta_single_ch["Delta"].append(d)
                delta_single_ch["beta"].append(beta_d)
                delta_single_ch["p-values"].append(p_vals_d)
                delta_single_ch["R^2"].append(r_2_d)
                
        if flag == 'DFA':
            df_single_ch_DFA=pd.DataFrame.from_dict(H_single_ch)
            
            # Create path for save data
            if os.path.isdir(f'{path_data}/DFA') == False:
                directory = 'DFA'
                parent_dir = f'{path_data}'
                path = os.path.join(parent_dir, directory) 
                os.makedirs(path, exist_ok= True) 

            # Save data 
            df_single_ch_DFA.to_csv(f"{path_data}/DFA/AvalancheWT_{subsequence}_jump_{jump}_dt_av_{dt_av}_N_av_{N_av}.csv", index=False)
            
        elif flag == 'DE':
            df_single_ch_DE=pd.DataFrame.from_dict(delta_single_ch)
            
            # Create path for save data
            if os.path.isdir(f'{path_data}/DE') == False:
                directory = 'DE'
                parent_dir = f'{path_data}'
                path = os.path.join(parent_dir, directory) 
                os.makedirs(path, exist_ok= True)  
            
            # Save data 
            df_single_ch_DE.to_csv(f"{path_data}/DE/AvalancheWT_{subsequence}_jump_{jump}_dt_av_{dt_av}_N_av_{N_av}.csv", index=False)
            
        else:
            df_single_ch_DFA=pd.DataFrame.from_dict(H_single_ch)
            
            # Create path for save data
            if os.path.isdir(f'{path_data}/DFA') == False:
                directory = 'DFA'
                parent_dir = f'{path_data}'
                path = os.path.join(parent_dir, directory) 
                os.makedirs(path, exist_ok= True) 

            # Save data 
            df_single_ch_DFA.to_csv(f"{path_data}/DFA/AvalancheWT_{subsequence}_jump_{jump}_dt_av_{dt_av}_N_av_{N_av}.csv", index=False)
            
            df_single_ch_DE=pd.DataFrame.from_dict(delta_single_ch)
            
            # Create path for save data
            if os.path.isdir(f'{path_data}/DE') == False:
                directory = 'DE'
                parent_dir = f'{path_data}'
                path = os.path.join(parent_dir, directory) 
                os.makedirs(path, exist_ok= True)  
            
            # Save data 
            df_single_ch_DE.to_csv(f"{path_data}/DE/AvalancheWT_{subsequence}_jump_{jump}_dt_av_{dt_av}_N_av_{N_av}.csv", index=False)
      
            
            
           
def single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, flag, H_exp):    
    H_single_ch={"Band":[], "Channel": [], "H": [], "beta": [], "p-values": [], "R^2": []}
    delta_single_ch={"Band":[], "Channel": [], "Delta": [], "beta": [], "p-values": [], "R^2": []}
    bands = pd.unique(data["Band"])
    channels= pd.unique(data["Channel"])
    for band in bands:
        sub_data_band = data[data["Band"] == band]
        for channel in channels:
            lista_ctrw = sub_data_band[sub_data_band["Channel"] == channel]            
            if type(lista_ctrw['Events'].iat[0]) == str:
                lista_ctrw = list(map(int,lista_ctrw['Events'].iat[0][1:-1].split(',')))
            else:
                lista_ctrw = lista_ctrw['Events'].iat[0]
            # lista = literal_eval(lista)
            label = f'subsequence_{subsequence} jump_{jump} band_{band} channel_{channel}'
            
            X = CTRW(sim_len,lista_ctrw,jump)
            
            if flag == 'DFA':
                h, p_vals, r_2, beta = DFA_channel_band(X, path_data, label, H_exp)

                H_single_ch["Band"].append(band)
                H_single_ch["Channel"].append(channel)
                H_single_ch["H"].append(h)
                H_single_ch["beta"].append(beta)
                H_single_ch["p-values"].append(p_vals)
                H_single_ch["R^2"].append(r_2)
                
            elif flag == 'DE':
                d, p_vals, r_2, beta = DE(X, path_data, label)
                
                delta_single_ch["Band"].append(band)
                delta_single_ch["Channel"].append(channel)
                delta_single_ch["Delta"].append(d)
                delta_single_ch["beta"].append(beta)
                delta_single_ch["p-values"].append(p_vals)
                delta_single_ch["R^2"].append(r_2)
                
            else:
                h, p_vals, r_2, beta = DFA_channel_band(X, path_data, label, H_exp)

                H_single_ch["Band"].append(band)
                H_single_ch["Channel"].append(channel)
                H_single_ch["H"].append(h)
                H_single_ch["beta"].append(beta)
                H_single_ch["p-values"].append(p_vals)
                H_single_ch["R^2"].append(r_2)
                
                d, p_vals_d, r_2_d, beta_d = DE(X, path_data, label)
                
                delta_single_ch["Band"].append(band)
                delta_single_ch["Channel"].append(channel)
                delta_single_ch["Delta"].append(d)  
                delta_single_ch["beta"].append(beta_d)
                delta_single_ch["p-values"].append(p_vals_d)
                delta_single_ch["R^2"].append(r_2_d)

    if flag == 'DFA':
        H_single_ch= pd.DataFrame.from_dict(H_single_ch)
        
        # Create path for save data
        if os.path.isdir(f'{path_data}/DFA') == False:
            directory = 'DFA'
            parent_dir = f'{path_data}'
            path = os.path.join(parent_dir, directory) 
            os.makedirs(path, exist_ok= True)
        
        # Save data 
        H_single_ch.to_csv(f"{path_data}/DFA/SingleCh_subsequence_{subsequence}_jump_{jump}.csv", index=False)
        
    elif flag == 'DE':
        delta_single_ch= pd.DataFrame.from_dict(delta_single_ch)
        
        # Create path for save data
        if os.path.isdir(f'{path_data}/DE') == False:
            directory = 'DE'
            parent_dir = f'{path_data}'
            path = os.path.join(parent_dir, directory) 
            os.makedirs(path, exist_ok= True) 
        
        # Save data 
        delta_single_ch.to_csv(f"{path_data}/DE/SingleCh_subsequence_{subsequence}_jump_{jump}.csv", index=False)
        
    else:
        H_single_ch= pd.DataFrame.from_dict(H_single_ch)
        
        # Create path for save data
        if os.path.isdir(f'{path_data}/DFA') == False:
            directory = 'DFA'
            parent_dir = f'{path_data}'
            path = os.path.join(parent_dir, directory) 
            os.makedirs(path, exist_ok= True)
        
        # Save data 
        H_single_ch.to_csv(f"{path_data}/DFA/SingleCh_subsequence_{subsequence}_jump_{jump}.csv", index=False)
        
        delta_single_ch= pd.DataFrame.from_dict(delta_single_ch)
        
        # Create path for save data
        if os.path.isdir(f'{path_data}/DE') == False:
            directory = 'DE'
            parent_dir = f'{path_data}'
            path = os.path.join(parent_dir, directory) 
            os.makedirs(path, exist_ok= True) 
        
        # Save data 
        delta_single_ch.to_csv(f"{path_data}/DE/SingleCh_subsequence_{subsequence}_jump_{jump}.csv", index=False)



################################################# DFA/DE ANALYZER #################################################
# def analyzer_DFA_DE(data, path_data, file_name, pipeline, Dt_av, N_av, sim_len, subsequence, subsequence_start, band_av, H_exp):
    
#     analysis_type = pipeline[10]
#     start_time = tm.time()

#     if (pipeline[9]== 'all'):
        
#         jumps=['AJ', 'SJ', 'SV']
#         c = 0
            
#         for jump in jumps:
                    
#             # DFA/DE analysis on avalanches WTs
#             if (analysis_type == "WT" or analysis_type == "wt"):
                
#                 if subsequence == 'full' and pipeline[7] == 'n' and pipeline[6] == 'n' and c == 0:
#                     c += 1
#                     data["Channel"]= data["Channel"].astype(np.int64)
#                     for i in range(data.shape[0]):
#                         data['Events'][i] = np.fromstring(data['Events'][i][1:-1], dtype=int, sep=',')
                
#                 for dt_av, n_av in zip(Dt_av, N_av):
                    
#                     if ((pipeline[8]=='DFA')):
                        
#                         if pipeline[7] == 'n':                                
#                             band_av = Waiting_Times(data, dt_av, n_av, subsequence, subsequence_start, path_data, file_name, pipeline)
                        
#                         print(f'Start DFA for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
#                         avalanche_WT_DFA_DE(band_av, dt_av, n_av, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                         print(f'Finish DFA for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
                        
#                     elif ((pipeline[8]=='DE')):
                        
#                         if pipeline[7] == 'n':                                
#                             band_av = Waiting_Times(data, dt_av, n_av, subsequence, subsequence_start, path_data, file_name, pipeline)
                        
#                         print(f'Start DE analysis for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
#                         avalanche_WT_DFA_DE(band_av, dt_av, n_av, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                         print(f'Finish DE analysis for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
                        
#                     else:
                        
#                         if pipeline[7] == 'n':                                
#                             band_av = Waiting_Times(data, dt_av, n_av, subsequence, subsequence_start, path_data, file_name, pipeline)
                        
#                         print(f'Start DFA and DE for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
#                         avalanche_WT_DFA_DE(band_av, dt_av, n_av, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                         print(f'Finish DFA and DE for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')                                
                                
#             # DFA/DE analysis on single channels
#             elif (analysis_type == "SC" or analysis_type == "sc"):
                
#                 if ((pipeline[8]=='DFA')):
                    
#                     print(f'Start DFA for jump type {jump}')
#                     single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                     print(f'Finish DFA for jump type {jump}')

#                 elif ((pipeline[8]=='DE')):
                    
#                     print(f'Start DE for jump type {jump}')
#                     single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                     print(f'Finish DE for jump type {jump}')
                        
#                 else :
                    
#                     print(f'Start DFA and DE for jump type {jump}')
#                     single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                     print(f'Finish DFA and DE for jump type {jump}')

#     else :
        
#         jump = pipeline[9]

#         # DFA/DE analysis on avalanches WTs
#         if (analysis_type == "WT" or analysis_type == "wt"):
            
#             if subsequence == 'full' and pipeline[7] == 'n' and pipeline[6] == 'n':
#                 data["Channel"]= data["Channel"].astype(np.int64)
#                 for i in range(data.shape[0]):
#                     data['Events'][i] = np.fromstring(data['Events'][i][1:-1], dtype=int, sep=',')
        
#             for dt_av, n_av in zip(Dt_av, N_av):
        
#                 if ((pipeline[8]=='DFA')):
                    
#                     if pipeline[7] == 'n':                            
#                         band_av = Waiting_Times(data, dt_av, n_av, subsequence, subsequence_start, path_data, file_name, pipeline)
                    
#                     print(f'Start DFA for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')  
#                     avalanche_WT_DFA_DE(band_av, dt_av, n_av, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                     print(f'Finish DFA for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
                    
#                 elif ((pipeline[8]=='DE')):
                    
#                     if pipeline[7] == 'n':                            
#                         band_av = Waiting_Times(data, dt_av, n_av, subsequence, subsequence_start, path_data, file_name, pipeline)
                    
#                     print(f'Start DE analysis for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
#                     avalanche_WT_DFA_DE(band_av, dt_av, n_av, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                     print(f'Finish DE analysis for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
                    
#                 else:
                    
#                     if pipeline[7] == 'n':                            
#                         band_av = Waiting_Times(data, dt_av, n_av, subsequence, subsequence_start, path_data, file_name, pipeline)
                    
#                     print(f'Start DFA and DE for dt_av = {dt_av}, N_av = {n_av} and jump type {jump}')
#                     avalanche_WT_DFA_DE(band_av, dt_av, n_av, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                     print(f'Finish DFA and DE for dt_av = {dt_av} and jump type {jump}')

#         # DFA/DE analysis on single channels
#         elif (analysis_type == "SC" or analysis_type == "sc"):
            
#             if ((pipeline[8]=='DFA')):
                
#                 print(f'Start DFA for jump type {jump}')
#                 single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                 print(f'Finish DFA for jump type {jump}')

#             elif ((pipeline[8]=='DE')):
                
#                 print(f'Start DE for jump type {jump}')
#                 single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                 print(f'Finish DE for jump type {jump}')
                    
#             else :
                
#                 print(f'Start DFA and DE for jump type {jump}')
#                 single_channel_DFA_DE(data, path_data, file_name, jump, sim_len, subsequence, pipeline[8], H_exp)
#                 print(f'Finish DFA and DE for jump type {jump}')

#     print("--- %s seconds for DFA/DE ---" % (tm.time() - start_time))