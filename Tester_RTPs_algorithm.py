"""
Tester for RTPs detection algorithm

"""

import numpy as np
import pandas as pd
import os
import importlib
import time as tm
import RTPs_package_new as na
import matplotlib.pyplot as plt
importlib.reload(na)
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
                    
############################### RTPs Detection ##################################

### PRELIMINARY STEPS
# Load pipeline parameters
# input_analyzer = 'Power_Law_WTs_different_noise_lfilter/input_analyzer.txt'

level_signal = 'True'
pos_derivative = 'True'

seed_list = [10]

for seed in seed_list:
    input_analyzer = 'Power_Law_WTs/input_analyzer.txt'
    f=open(input_analyzer, 'r')
    pipeline=np.loadtxt(f, dtype='str')
    f.close()

    start_time_code = tm.time()

    ### EVENT DETECTION: YES
    if (pipeline[0]=='y'):
        
        # Load parameters for event detection
        RTP_input_parameters = pipeline[1]
        
        eeg_name_list = pipeline[2].split('\\')
        
        for eeg_name in eeg_name_list:
            
            H_exp = 0.5
            
            n_impulses = int(eeg_name.split('/')[-1].split('_')[(eeg_name.split('/')[-1].split('_')).index('impulses') + 1])
            
            if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                
                mu = eeg_name.split('/')[-1].split('_')[(eeg_name.split('/')[-1].split('_')).index('mu') + 1] 
                T = eeg_name.split('/')[-1].split('_')[(eeg_name.split('/')[-1].split('_')).index('T') + 1]
                
                mu_num = float(mu)
                
                if 1 < mu_num < 2:
                    H_exp = mu_num/2
                elif 2 <= mu_num < 3:
                    H_exp = 2 - (mu_num/2)
                else:
                    H_exp = 0.5
            
            # Read eeg_data
            eeg_data_name = f'{eeg_name}.txt'
            
            f=open(eeg_data_name, 'r')
            eeg_data = np.loadtxt(f, dtype='float')
            f.close() 
            
            # Set result directory
            tmp=eeg_name.split('/')
            path_data='/'.join(tmp[:-1])
            eeg_file_name=tmp[len(tmp)-1]
            
            if os.path.isdir(f'{path_data}/{eeg_file_name}_Event_Detection') == False:
                result_directory = f'{eeg_file_name}_Event_Detection'
                working_directory = f'{path_data}'
                path = os.path.join(working_directory, result_directory)
                os.mkdir(path)
                print("Directory '% s' created" % result_directory)
                
            if os.path.isdir(f'{path_data}/output') == False:
                result_directory = 'output'
                working_directory = f'{path_data}'
                path = os.path.join(working_directory, result_directory)
                os.mkdir(path)
                print("Directory '% s' created" % result_directory)
                
            # single-channel event detection
            if level_signal == 'True':
                files_name_list, TS_name_list, LS_name_list, der_name_list = na.RTP_extraction(eeg_name, RTP_input_parameters, level_signal = level_signal, pos_der = pos_derivative)
            else:
                files_name_list, TS_name_list, der_name_list = na.RTP_extraction(eeg_name, RTP_input_parameters, level_signal = level_signal, pos_der = pos_derivative)
            
            ### CHECK IF THE RTPs ARE CORRECT
            
            if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                real_RTPs_name = f'{path_data}/impulse_series_T_{T}_mu_{mu}_n_impulses_{n_impulses}_seed_{seed}.txt'
            else:
                real_RTPs_name = f'{path_data}/impulse_series_r_p_1.0_n_impulses_{n_impulses}_seed_{seed}.txt'                

            f=open(real_RTPs_name, 'r')
            series =np.loadtxt(f, dtype='float')
            f.close() 
            
            impulse_series = series

            real_RTPs = np.concatenate(np.argwhere(series))
            
            np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/real_RTPs.txt', real_RTPs)
            
            real_WTs = np.diff(real_RTPs)

            # Histogram of real WTs
            occurrences_real_wt, real_wt_hist = np.histogram(real_WTs, bins = int(max(real_WTs)))
            
            np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/real_wt_histogram.txt', [real_wt_hist[:-1], occurrences_real_wt])
            
            # DFA/DE on real events
            jump = 'AJ'
            analysis_type = 'both'
            X = na.CTRW(len(eeg_data), real_RTPs, jump)
            H_real = na.DFA_channel_band(X, f'{path_data}/{eeg_file_name}_Event_Detection', f'jump_{jump}', H_exp)
            delta_real = na.DE(X, f'{path_data}/{eeg_file_name}_Event_Detection', f'jump_{jump}')
            
            p_value_list = []
            estimated_RTPs_len_list = []
            estimated_RTPs_len_list.append('# Real events:')
            estimated_RTPs_len_list.append(len(real_RTPs))
            missed_RTPs_list = []
            fp_RTPs_list = []
            
            if level_signal == 'True':
                
                for file_name, TS_name, LS_name, der_name in zip(files_name_list, TS_name_list, LS_name_list, der_name_list):
                    
                    estimated_RTPs_dataframe = pd.read_csv(f'{path_data}/{eeg_file_name}_Event_Detection/{file_name}.csv')

                    estimated_RTPs = np.asarray(list(map(int,estimated_RTPs_dataframe.loc[0, 'Events'][1:-1].split(','))))
                    
                    np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/estimated_RTPs_{file_name}.txt', estimated_RTPs)
                    
                    estimated_RTPs_len_list.append(f'# {file_name} founded events:')
                    estimated_RTPs_len_list.append(len(estimated_RTPs))
                    
                    TS_path = f'{path_data}/{eeg_file_name}_Event_Detection/{TS_name}.txt'

                    f=open(TS_path, 'r')
                    TS =np.loadtxt(f, dtype='float')
                    f.close() 
                    
                    LS_path = f'{path_data}/{eeg_file_name}_Event_Detection/{LS_name}.txt'

                    f=open(LS_path, 'r')
                    LS =np.loadtxt(f, dtype='float')
                    f.close() 
                    
                    der_path = f'{path_data}/{eeg_file_name}_Event_Detection/{der_name}.txt'

                    f=open(der_path, 'r')
                    der =np.loadtxt(f, dtype='float')
                    f.close() 
                    
                    ## Calculate estimated WTs
                    estimated_WTs = np.diff(estimated_RTPs)
                    
                    ## Mann-Whitney test
                    U1, p_NORM = mannwhitneyu(real_WTs[(real_WTs>30)], estimated_WTs[(estimated_WTs>30)], use_continuity = False, method="auto")
                    print(f'Mann-Whitney p_value between {file_name} and {eeg_file_name} is: {p_NORM}')
                    p_value_list.append(f'# {file_name} WTs Mann-Whitney p_value:')
                    p_value_list.append(p_NORM)
                    
                    
                    ## Kolmogorov-Smirnov test
                    r = ks_2samp(real_WTs[(real_WTs>30)], estimated_WTs[(estimated_WTs>30)], method="auto")
                    print(f'Kolmogorov-Smirnov p_value between {file_name} and {eeg_file_name} is: {r.pvalue}')
                    p_value_list.append(f'# {file_name} WTs Kolmogorov-Smirnov p_value:')
                    p_value_list.append(r.pvalue)
                    
                    
                    ## Calculate estimated WTs distribution                    
                    occurrences_estimated_wt, estimated_wt_hist = np.histogram(estimated_WTs, bins = int(max(estimated_WTs)))
                    
                    # Plot WTs histograms
                    plt.figure(figsize=(13,12))
                    plt.subplot(311)
                    plt.plot(real_wt_hist[:-1], occurrences_real_wt, '+')
                    if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                        plt.xscale('log')
                        plt.yscale('log')
                    else:
                        plt.yscale('log')
                    # plt.xlim([0, 500])
                    # plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of real WTs')
                    
                    plt.subplot(312)
                    plt.plot(estimated_wt_hist[:-1], occurrences_estimated_wt, '+')
                    if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                        plt.xscale('log')
                        plt.yscale('log')
                    else:
                        plt.yscale('log')
                    # plt.xlim([0, 800])
                    # plt.ylim([0, 30])
                    # plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of estimated WTs')
                    
                    plt.subplot(313)
                    plt.plot(estimated_wt_hist[:-1], occurrences_estimated_wt, '+')
                    if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                        plt.xscale('log')
                        plt.yscale('log')
                    else:
                        plt.yscale('log')
                    plt.xlim([0, max(real_wt_hist[:-1])])
                    plt.ylim([min(occurrences_real_wt), max(occurrences_real_wt)])
                    plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of estimated WTs (Zoom)')
                    
                    np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/estimated_wt_histograms_RTPs_{file_name}.txt', [estimated_wt_hist[:-1], occurrences_estimated_wt])
                    plt.savefig(f'{path_data}/{eeg_file_name}_Event_Detection/result_wt_histograms_RTPs_{file_name}.png', dpi = 600)
                    
                    plt.show()
                    
                    
                    t = np.arange(0, len(impulse_series))
                    
                                
                    ## Plot real events vs estimated events
                    plt.figure(figsize=(18,17))
                    plt.subplot(411)
                    plt.plot(t, eeg_data)
                    plt.vlines(real_RTPs, ymin = -10, ymax = 10)
                    plt.xlim([21000, 25000])
                    plt.ylim([min(eeg_data), max(eeg_data)])
                    # plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    plt.title('EEG signal with real RTPs')
                    
                    plt.subplot(412)
                    plt.plot(t, eeg_data)
                    plt.vlines(estimated_RTPs, ymin = -10, ymax = 10)
                    plt.xlim([21000, 25000])
                    plt.ylim([min(eeg_data), max(eeg_data)])
                    # plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    plt.title('EEG signal with estimated RTPs')
                    
                    plt.subplot(413)
                    plt.plot(TS, 'r-')
                    plt.plot(LS, 'b-')
                    # plt.plot(der)
                    # plt.vlines(estimated_RTPs, ymin = -1, ymax = 1)
                    plt.xlim([21000, 25000])
                    # plt.ylim([-0.15, 0.15])
                    # plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    plt.title('Level and Test signals with estimated RTPs')
                    
                    plt.subplot(414)
                    # plt.plot(TS, 'r-')
                    # plt.plot(LS, 'b-')
                    plt.plot(der)
                    # plt.vlines(estimated_RTPs, ymin = -1, ymax = 1)
                    plt.xlim([21000, 25000])
                    # plt.ylim([-0.2, 0.4])
                    plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    # plt.title('Estimated Derivative of the signal S = TS - LS')
                    plt.title('Estimated Derivative of the TS')
                    
                    plt.savefig(f'{path_data}/{eeg_file_name}_Event_Detection/result_RTPs_{file_name}.png', dpi = 600)
                    
                    plt.show()
                    
                    
                    
                    # DFA & DE on Events
                    # Create folder where save results
                    result_number = 1                
                    while os.path.exists(f"{path_data}/{eeg_file_name}_Event_Detection/{file_name}_IDC_{result_number}") == True:
                        result_number += 1
                    
                    result_directory = f'{file_name}_IDC_{result_number}'
                    working_directory = f'{path_data}/{eeg_file_name}_Event_Detection'
                    path = os.path.join(working_directory, result_directory)
                    os.mkdir(path)
                    print("Directory '% s' created" % result_directory)
                    
                    # Save the path for the results
                    path_results = f"{path_data}/{eeg_file_name}_Event_Detection/{file_name}_IDC_{result_number}"
                    
                    # DFA/DE on estimated RTPs
                    # band_av = 0
                    # n_split = 'full'
                    # start_time = 0
                    # dt_av = 1
                    # N_av = 100
                    # sim_len = len(eeg_data)
                    # na.analyzer_DFA_DE(estimated_RTPs_dataframe, path_results, file_name, pipeline, dt_av, N_av, sim_len, n_split, start_time, band_av, H_exp)
                    
                    # jump = 'AJ'
                    # X = na.CTRW(len(eeg_data), real_RTPs, jump)
                    # H_real = na.DFA_channel_band(X, f'{path_data}/{eeg_file_name}_Event_Detection', f'jump_{jump}', H_exp)
                    # delta_real = na.DE(X, f'{path_data}/{eeg_file_name}_Event_Detection', f'jump_{jump}')

                    na.single_channel_DFA_DE(estimated_RTPs_dataframe, path_results, file_name, jump, len(eeg_data), 'full', analysis_type, H_exp)                    
                    
                    
                    # Check if there are some RTPs perfectly founded
                    real_est_RTPs_eps_0 = []

                    new_real_RTPs = []

                    new_estimated_RTPs = estimated_RTPs

                    count_eps = []

                    count = 0

                    for i in real_RTPs:
                        if i in estimated_RTPs:
                            count += 1
                            real_est_RTPs_eps_0.append(i)
                            new_estimated_RTPs= np.delete(new_estimated_RTPs, np.where(new_estimated_RTPs==i))
                        else:
                            new_real_RTPs.append(i)
                            
                    count_eps.append(count)

                    real_est_RTPs_list = []
                    
                    real_est_RTPs_list.append(list(['0', real_est_RTPs_eps_0]))

                    # Check if there are some RTPs founded with an eps of error
                    eps = 1

                    while True:
                        count = 0
                        # globals()['real_est_RTPs_eps_'+str(eps)] = []
                        real_est_RTPs_eps = []
                        
                        new_real_RTPs_temp = new_real_RTPs
                        
                        new_estimated_RTPs_temp  = new_estimated_RTPs
                        
                        for i in new_real_RTPs:
                            
                            element_i = np.arange(i - eps, i + eps + 1, 1)

                            control_range = np.isin(element_i, new_estimated_RTPs)    
                            
                            if True in control_range:
                                count += 1
                                new_real_RTPs_temp = np.delete(new_real_RTPs_temp, np.where(new_real_RTPs_temp==i))
                                real_est_RTPs_eps.append(i)
                                for idx, j in enumerate(element_i):
                                    if control_range[idx] == True:
                                        new_estimated_RTPs_temp = np.delete(new_estimated_RTPs_temp, np.where(new_estimated_RTPs_temp==j))
                                        break    
                        
                        new_estimated_RTPs = new_estimated_RTPs_temp
                        
                        new_real_RTPs = new_real_RTPs_temp
                        
                        if len(new_real_RTPs) == 0 or len(new_estimated_RTPs) == 0 or count == 0:
                            break

                        real_est_RTPs_list.append(list([f'{eps}', real_est_RTPs_eps]))
                        
                        eps += 1
                        
                        count_eps.append(count)
                        
                    missed_RTPs_list.append(f'# {file_name} missed real RTPs:')
                    missed_RTPs_list.append(new_real_RTPs)
                    missed_RTPs_list.append(len(new_real_RTPs))
                    
                    fp_RTPs_list.append(f'# {file_name} false positive RTPs:')
                    fp_RTPs_list.append(new_estimated_RTPs)
                    fp_RTPs_list.append(len(new_estimated_RTPs))
                    
                    real_est_RTPs = pd.DataFrame(real_est_RTPs_list).T
                    
                    real_est_RTPs.columns = real_est_RTPs.iloc[0]
                    
                    real_est_RTPs = real_est_RTPs[1:]

                    real_est_RTPs.to_csv(f"{path_data}/{eeg_file_name}_Event_Detection/real_founded_RTPs_vs_eps_{file_name}.csv", index=False)

                    plt.figure(figsize=(8,7))
                    eps_list = np.arange(0, eps, 1)
                    plt.plot(eps_list, count_eps, 'o-')
                    plt.xlabel(r'$\epsilon$')
                    plt.ylabel('counts')
                    plt.title('Real RTPs founded vs. Tollerance')
                    plt.savefig(f'{path_data}/{eeg_file_name}_Event_Detection/count_real_RTPs_vs_eps_{file_name}.png', dpi = 600)
                    np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/count_real_RTPs_vs_eps_{file_name}.txt', [eps_list, count_eps])
                    plt.show()
                    
                # np.savetxt(f'{path_data}/p_values_{eeg_file_name}.txt', np.array(p_value_list))
                with open(f'{path_data}/output/p_values_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in p_value_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
                with open(f'{path_data}/output/real_and_estimated_RTPs_len_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in estimated_RTPs_len_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
                with open(f'{path_data}/output/missed_real_RTPs_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in missed_RTPs_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
                with open(f'{path_data}/output/false_positive_RTPs_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in fp_RTPs_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
            else:
                for file_name, TS_name, der_name in zip(files_name_list, TS_name_list, der_name_list):
                    
                    estimated_RTPs_dataframe = pd.read_csv(f'{path_data}/{eeg_file_name}_Event_Detection/{file_name}.csv')

                    estimated_RTPs = np.asarray(list(map(int,estimated_RTPs_dataframe.loc[0, 'Events'][1:-1].split(','))))
                    
                    np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/estimated_RTPs_{file_name}.txt', estimated_RTPs)
                    
                    estimated_RTPs_len_list.append(f'# {file_name} founded events:')
                    estimated_RTPs_len_list.append(len(estimated_RTPs))
                    
                    TS_path = f'{path_data}/{eeg_file_name}_Event_Detection/{TS_name}.txt'

                    f=open(TS_path, 'r')
                    TS =np.loadtxt(f, dtype='float')
                    f.close()
                    
                    der_path = f'{path_data}/{eeg_file_name}_Event_Detection/{der_name}.txt'

                    f=open(der_path, 'r')
                    der =np.loadtxt(f, dtype='float')
                    f.close() 
                    
                    ## Calculate estimated WTs
                    estimated_WTs = np.diff(estimated_RTPs)
                    
                    # ## Mann-Whitney test
                    # U1, p_NORM = mannwhitneyu(real_WTs, estimated_WTs, use_continuity = False, method="auto")
                    # print(f'Mann-Whitney p_value between {file_name} and {eeg_file_name} is: {p_NORM}')
                    # p_value_list.append(f'{file_name} WTs Mann-Whitney p_value:')
                    # p_value_list.append(p_NORM)
                    
                    
                    # ## Kolmogorov-Smirnov test
                    # r = ks_2samp(real_WTs, estimated_WTs, method="auto")
                    # print(f'Kolmogorov-Smirnov p_value between {file_name} and {eeg_file_name} is: {r.pvalue}')
                    # p_value_list.append(f'{file_name} WTs Kolmogorov-Smirnov p_value:')
                    # p_value_list.append(r.pvalue)
                    
                    ## Mann-Whitney test
                    U1, p_NORM = mannwhitneyu(real_WTs[(real_WTs>30)], estimated_WTs[(estimated_WTs>30)], use_continuity = False, method="auto")
                    print(f'Mann-Whitney p_value between {file_name} and {eeg_file_name} is: {p_NORM}')
                    p_value_list.append(f'# {file_name} WTs Mann-Whitney p_value:')
                    p_value_list.append(p_NORM)
                    
                    
                    ## Kolmogorov-Smirnov test
                    r = ks_2samp(real_WTs[(real_WTs>30)], estimated_WTs[(estimated_WTs>30)], method="auto")
                    print(f'Kolmogorov-Smirnov p_value between {file_name} and {eeg_file_name} is: {r.pvalue}')
                    p_value_list.append(f'# {file_name} WTs Kolmogorov-Smirnov p_value:')
                    p_value_list.append(r.pvalue)
                    
                    
                    ## Calculate estimated WTs distribution                    
                    occurrences_estimated_wt, estimated_wt_hist = np.histogram(estimated_WTs, bins = int(max(estimated_WTs)))
                    
                    # Plot WTs histograms
                    plt.figure(figsize=(13,12))
                    plt.subplot(311)
                    plt.plot(real_wt_hist[:-1], occurrences_real_wt, '+')
                    if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                        plt.xscale('log')
                        plt.yscale('log')
                    else:
                        plt.yscale('log')
                    # plt.xlim([0, 500])
                    # plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of real WTs')
                    
                    plt.subplot(312)
                    plt.plot(estimated_wt_hist[:-1], occurrences_estimated_wt, '+')
                    if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                        plt.xscale('log')
                        plt.yscale('log')
                    else:
                        plt.yscale('log')
                    # plt.xlim([0, 800])
                    # plt.ylim([0, 30])
                    # plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of estimated WTs')
                    
                    plt.subplot(313)
                    plt.plot(estimated_wt_hist[:-1], occurrences_estimated_wt, '+')
                    if input_analyzer.split('/')[0].split('_')[0] == 'Power':
                        plt.xscale('log')
                        plt.yscale('log')
                    else:
                        plt.yscale('log')
                    plt.xlim([0, max(real_wt_hist[:-1])])
                    plt.ylim([min(occurrences_real_wt), max(occurrences_real_wt)])
                    plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of estimated WTs (Zoom)')
                    
                    np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/estimated_wt_histograms_RTPs_{file_name}.txt', [estimated_wt_hist[:-1], occurrences_estimated_wt])
                    plt.savefig(f'{path_data}/{eeg_file_name}_Event_Detection/result_wt_histograms_RTPs_{file_name}.png', dpi = 600)
                    
                    plt.show()
                    
                    
                    t = np.arange(0, len(impulse_series))
                    
                                
                    ## Plot real events vs estimated events
                    plt.figure(figsize=(18,17))
                    plt.subplot(411)
                    plt.plot(t, eeg_data)
                    plt.vlines(real_RTPs, ymin = -10, ymax = 10)
                    plt.xlim([21000, 25000])
                    plt.ylim([min(eeg_data), max(eeg_data)])
                    # plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    plt.title('EEG signal with real RTPs')
                    
                    plt.subplot(412)
                    plt.plot(t, eeg_data)
                    plt.vlines(estimated_RTPs, ymin = -10, ymax = 10)
                    plt.xlim([21000, 25000])
                    plt.ylim([min(eeg_data), max(eeg_data)])
                    # plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    plt.title('EEG signal with estimated RTPs')
                    
                    plt.subplot(413)
                    plt.plot(TS, 'r-')
                    # plt.plot(LS, 'b-')
                    # plt.plot(der)
                    # plt.vlines(estimated_RTPs, ymin = -1, ymax = 1)
                    plt.xlim([21000, 25000])
                    # plt.ylim([-0.15, 0.15])
                    # plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    plt.title('Test signal with estimated RTPs')
                    
                    plt.subplot(414)
                    # plt.plot(TS, 'r-')
                    # plt.plot(LS, 'b-')
                    plt.plot(der)
                    # plt.vlines(estimated_RTPs, ymin = -1, ymax = 1)
                    plt.xlim([21000, 25000])
                    # plt.ylim([-0.2, 0.4])
                    plt.xlabel('time')
                    plt.ylabel('signal amplitude')
                    # plt.title('Estimated Derivative of the signal S = TS - LS')
                    plt.title('Estimated Derivative of the TS')
                    
                    plt.savefig(f'{path_data}/{eeg_file_name}_Event_Detection/result_RTPs_{file_name}.png', dpi = 600)
                    
                    plt.show()
                    
                    
                    
                    # DFA & DE on Events
                    # Create folder where save results
                    result_number = 1                
                    while os.path.exists(f"{path_data}/{eeg_file_name}_Event_Detection/{file_name}_IDC_{result_number}") == True:
                        result_number += 1
                    
                    result_directory = f'{file_name}_IDC_{result_number}'
                    working_directory = f'{path_data}/{eeg_file_name}_Event_Detection'
                    path = os.path.join(working_directory, result_directory)
                    os.mkdir(path)
                    print("Directory '% s' created" % result_directory)
                    
                    # Save the path for the results
                    path_results = f"{path_data}/{eeg_file_name}_Event_Detection/{file_name}_IDC_{result_number}"
                    
                    # DFA/DE on estimated RTPs
                    # band_av = 0
                    # n_split = 'full'
                    # start_time = 0
                    # dt_av = 1
                    # N_av = 100
                    # sim_len = len(eeg_data)
                    # na.analyzer_DFA_DE(estimated_RTPs_dataframe, path_results, file_name, pipeline, dt_av, N_av, sim_len, n_split, start_time, band_av, H_exp)
                    
                    na.single_channel_DFA_DE(estimated_RTPs_dataframe, path_results, file_name, jump, len(eeg_data), 'full', analysis_type, H_exp)
                    
                    # Check if there are some RTPs perfectly founded
                    real_est_RTPs_eps_0 = []

                    new_real_RTPs = []

                    new_estimated_RTPs = estimated_RTPs

                    count_eps = []

                    count = 0

                    for i in real_RTPs:
                        if i in estimated_RTPs:
                            count += 1
                            real_est_RTPs_eps_0.append(i)
                            new_estimated_RTPs= np.delete(new_estimated_RTPs, np.where(new_estimated_RTPs==i))
                        else:
                            new_real_RTPs.append(i)
                            
                    count_eps.append(count)

                    real_est_RTPs_list = []
                    
                    real_est_RTPs_list.append(list(['0', real_est_RTPs_eps_0]))

                    # Check if there are some RTPs founded with an eps of error
                    eps = 1

                    while True:
                        count = 0
                        # globals()['real_est_RTPs_eps_'+str(eps)] = []
                        real_est_RTPs_eps = []
                        
                        new_real_RTPs_temp = new_real_RTPs
                        
                        new_estimated_RTPs_temp  = new_estimated_RTPs
                        
                        for i in new_real_RTPs:
                            
                            element_i = np.arange(i - eps, i + eps + 1, 1)

                            control_range = np.isin(element_i, new_estimated_RTPs)    
                            
                            if True in control_range:
                                count += 1
                                new_real_RTPs_temp = np.delete(new_real_RTPs_temp, np.where(new_real_RTPs_temp==i))
                                real_est_RTPs_eps.append(i)
                                for idx, j in enumerate(element_i):
                                    if control_range[idx] == True:
                                        new_estimated_RTPs_temp = np.delete(new_estimated_RTPs_temp, np.where(new_estimated_RTPs_temp==j))
                                        break    
                        
                        new_estimated_RTPs = new_estimated_RTPs_temp
                        
                        new_real_RTPs = new_real_RTPs_temp
                        
                        if len(new_real_RTPs) == 0 or len(new_estimated_RTPs) == 0 or count == 0:
                            break
                        
                        real_est_RTPs_list.append(list([f'{eps}', real_est_RTPs_eps]))
                        
                        eps += 1
                        
                        count_eps.append(count)
                        
                    missed_RTPs_list.append(f'# {file_name} missed real RTPs:')
                    missed_RTPs_list.append(new_real_RTPs)
                    missed_RTPs_list.append(len(new_real_RTPs))
                    
                    fp_RTPs_list.append(f'# {file_name} false positive RTPs:')
                    fp_RTPs_list.append(new_estimated_RTPs)
                    fp_RTPs_list.append(len(new_estimated_RTPs))
                    
                    real_est_RTPs = pd.DataFrame(real_est_RTPs_list).T
                    
                    real_est_RTPs.columns = real_est_RTPs.iloc[0]
                    
                    real_est_RTPs = real_est_RTPs[1:]

                    real_est_RTPs.to_csv(f"{path_data}/{eeg_file_name}_Event_Detection/real_founded_RTPs_vs_eps_{file_name}.csv", index=False)

                    plt.figure(figsize=(8,7))
                    eps_list = np.arange(0, eps, 1)
                    plt.plot(eps_list, count_eps, 'o-')
                    plt.xlabel(r'$\epsilon$')
                    plt.ylabel('counts')
                    plt.title('Real RTPs founded vs. Tollerance')
                    plt.savefig(f'{path_data}/{eeg_file_name}_Event_Detection/count_real_RTPs_vs_eps_{file_name}.png', dpi = 600)
                    np.savetxt(f'{path_data}/{eeg_file_name}_Event_Detection/count_real_RTPs_vs_eps_{file_name}.txt', [eps_list, count_eps])
                    plt.show()
                    
                # np.savetxt(f'{path_data}/p_values_{eeg_file_name}.txt', np.array(p_value_list))
                with open(f'{path_data}/output/p_values_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in p_value_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
                with open(f'{path_data}/output/real_and_estimated_RTPs_len_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in estimated_RTPs_len_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
                with open(f'{path_data}/output/missed_real_RTPs_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in missed_RTPs_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
                
                with open(f'{path_data}/output/false_positive_RTPs_{eeg_file_name}.txt', 'w') as f:
                    
                    for item in fp_RTPs_list:
                        
                        line = str(item) + '\n'
                        
                        f.write(line)
                f.close()
    
    
print("--- %s seconds for Test_RTPs_algorithm.py ---" % (tm.time() - start_time_code))

    
    


