"""
Example for RTP-Distribution-Finder

"""

import numpy as np
import pandas as pd
import importlib
import RTP_finder_package as na
import matplotlib.pyplot as plt
importlib.reload(na)
                    
############################### RTPs Detection ##################################

data_name = ['./data.csv']

fs = 100
band_lst = [[-4,-2], [8,12]]
percentile_lst = [90, 95]
n_derivative_lst = [5, 10]

event_detection_file_name_lst = na.RTP_extraction(data_name, fs, bands = band_lst, percentiles = percentile_lst, n_der = n_derivative_lst)

data = pd.read_csv(data_name[0]).to_numpy()
t = np.linspace(0, len(data)/fs, len(data))

for event_detection_file_name in event_detection_file_name_lst:

    estimated_RTPs_dataframe = pd.read_csv(f'{event_detection_file_name}')

    estimated_RTPs = np.asarray(list(map(int,estimated_RTPs_dataframe.loc[0, 'Events'][1:-1].split(','))))
    
    estimated_RTPs_for_plot = []
    for i in estimated_RTPs:
        estimated_RTPs_for_plot.append(i/fs)

    ## Plot Events Detected
    plt.figure(figsize=(18,17))
    plt.vlines(estimated_RTPs_for_plot, ymin = min(data), ymax = max(data), colors='red')
    plt.plot(t, data)
    plt.xlim([70, 100])
    # plt.ylim([min(data), max(data)])
    plt.xlabel('time')
    plt.ylabel('signal amplitude')
    plt.title('Signal with estimated RTPs')
    plt.show()

    
    


