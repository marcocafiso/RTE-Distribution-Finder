"""
Code to simulate an EEG signal with Exponential WTs distribution 
or Uniform WTs distribution or Power-Law WTs distribution

"""

from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import time as tm
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


##################### MAIN OF PROGRAM #########################

## Set Parameters

start_time_code = tm.time()

# Open text file (.txt) with the parameters for simulation
par = []
with open('parameters_eeg_simulator_bands.txt', 'r') as file:
    lines = [line.rstrip() for line in file if line.rstrip()]
    for line in lines:
        if not line.startswith('#'):
            par.append(line)

# Set WTs_distribution
arr = [int(i.strip()) for i in par[0][1:-1].split(",")]
WTs_distribution_list = np.array(arr)

# Set parameter Ts
arr = [float(i.strip()) for i in par[1][1:-1].split(",")]
Ts = np.array(arr)[0]

# Set noise variance
arr = [float(i.strip()) for i in par[2][1:-1].split(",")]
perc_sigma2_list = np.array(arr)

# Set noise mean
arr = [float(i.strip()) for i in par[3][1:-1].split(",")]
average_noise = np.array(arr)[0]

# Set alpha
arr = [float(i.strip()) for i in par[4][1:-1].split(",")]
alpha = np.array(arr)[0]

# Set gamma
arr = [float(i.strip()) for i in par[5][1:-1].split(",")]
gamma = np.array(arr)[0]

# Set omega_0_min list
arr = [float(i.strip()) for i in par[6][1:-1].split(",")]
omega_0_min_list = np.asarray(arr)

# Set omega_0_max list
arr = [float(i.strip()) for i in par[7][1:-1].split(",")]
omega_0_max_list = np.asarray(arr)

# Set amplitude range of impulses
arr = [float(i.strip()) for i in par[8][1:-1].split(",")]
amp_impulses = np.asarray(arr)
amp_min = amp_impulses[0]
amp_max = amp_impulses[1]

# Set number of impulses
arr = [int(i.strip()) for i in par[9][1:-1].split(",")]
n_impulse = np.asarray(arr)[0]

# Set r_p
arr = [float(i.strip()) for i in par[10][1:-1].split(",")]
r_p = np.asarray(arr)[0]

# Set mu
arr = [float(i.strip()) for i in par[11][1:-1].split(",")]
mu_list = np.asarray(arr)

# Set tau
arr = [float(i.strip()) for i in par[12][1:-1].split(",")]
tau = np.asarray(arr)[0]

# Set seed values
arr = [int(i.strip()) for i in par[13][1:-1].split(",")]
seed_values = np.asarray(arr)

# Clean temporary variables
del lines, line, par, arr

for seed_value in seed_values:
                         
        ## Path for save results
        mother_path = os.getcwd()
      
        ## Set seed
        np.random.seed(seed_value)
        
        for WTs_distribution in WTs_distribution_list:

            if WTs_distribution == 0:
                
                print(f'Start simulation Exp WTs distribution, seed {seed_value}')
                
                # Create path for saves
                if os.path.isdir(f'{mother_path}/output/Exp_WTs') == False:
                    directory = 'Exp_WTs'
                    parent_dir = f'{mother_path}/output'
                    path = os.path.join(parent_dir, directory) 
                    os.makedirs(path, exist_ok= True)
                    
                path_data = f'{mother_path}/output/Exp_WTs'
                
                # Impulse Timing
                impulse_WTs = []

                for i in range(0, n_impulse+1):
                    
                    impulse_WTs.append(-(1/r_p)*(np.log(np.random.uniform(0,1, size = 1))))
                        
                impulse_WTs = np.concatenate(impulse_WTs)

                plt.figure(figsize=(8,7))
                occurrences, wt = np.histogram(impulse_WTs, bins = int(max(impulse_WTs)))
                plt.plot(wt[:-1], occurrences, '+')
                # plt.xscale('log')
                plt.yscale('log')
                # plt.xlim([0, 500])
                plt.xlabel('WT')
                plt.ylabel('# of occurrences')
                plt.title('Histogram of WT')
                np.savetxt(f'{path_data}/histogram_WTs_r_p_{r_p}_seed_{seed_value}.txt', [wt[:-1], occurrences])
                plt.savefig(f'{path_data}/histogram_WTs_r_p_{r_p}_seed_{seed_value}.png', dpi = 600)
                plt.close()

                impulse_timing_full = np.cumsum(impulse_WTs)

                impulse_timing = impulse_timing_full[:-1]

                # Define sampling grid
                t_max = max(impulse_timing_full)

                t = np.arange(0, t_max, step = Ts)

                np.savetxt(f'{path_data}/sampling_grid_seed_{seed_value}.txt', t)

                # Impulse Amplitude
                impulse_amplitudes = np.random.uniform(amp_min, amp_max, n_impulse)
                
                # Impulse Series
                impulse_series = np.zeros(len(t))

                # Find the indices in t that are closest to the impulse timings
                # indices = np.abs(np.subtract.outer(t, impulse_timing)).argmin(axis=0)
                indices = []
                for temp in impulse_timing:
                    indices.append(np.abs(t - temp).argmin())

                # Assign the impulse amplitudes at these indices
                impulse_series[indices] = impulse_amplitudes
                    
                # Plot Impulse series
                plt.figure(figsize=(8,7))
                plt.stem(t, impulse_series, markerfmt= ' ')
                plt.title('Impulse Series')
                plt.xlabel('Time')
                plt.ylabel('Impulse Amplitude')
                np.savetxt(f'{path_data}/impulse_series_r_p_{r_p}_n_impulses_{n_impulse}_seed_{seed_value}.txt', impulse_series)
                plt.savefig(f'{path_data}/impulse_series_r_p_{r_p}_n_impulses_{n_impulse}_seed_{seed_value}.png', dpi = 600)
                plt.close()

                for omega_0_min, omega_0_max in zip(omega_0_min_list, omega_0_max_list):
                    
                    print(f'omega_0 range = [{omega_0_min}, {omega_0_max}]')    
                    
                    final_signal_without_noise = np.zeros(len(t))
                    
                    for omega_0 in np.arange(omega_0_min, omega_0_max, 1):
                        
                        if omega_0 == omega_0_min + (omega_0_max - omega_0_min)//4:
                            print('25%')
                            
                        elif omega_0 == omega_0_min + (omega_0_max - omega_0_min)//2:
                            
                            print('50%')
                            
                        elif omega_0 == omega_0_min + ((omega_0_max - omega_0_min)//4 + (omega_0_max - omega_0_min)//2):
                            
                            print('75%')
                            
                        ## Convolution of Impulse series with the Impulse response
                        if math.isnan(alpha):
                            
                            # Underdumped oscillator
                            omega_1 = (1/2)*np.sqrt(4*(omega_0**2) - (gamma**2))
                            cost = -2/(4*(omega_1**3) + (omega_1*(gamma**2)))                    
                            b = [cost*(math.exp(-(gamma/2)*Ts)*omega_1*2*math.cos(omega_1*Ts) + math.exp(-(gamma/2)*Ts)*gamma*math.sin(omega_1*Ts) - 2*omega_1), cost*(-math.exp(-gamma*Ts)*2*omega_1 + math.exp(-(gamma/2)*Ts)*omega_1*2*math.cos(omega_1*Ts) - math.exp(-(gamma/2)*Ts)*gamma*math.sin(omega_1*Ts))]
                            a = [1, -2*math.exp(-(gamma/2)*Ts)*math.cos(omega_1*Ts), math.exp(-gamma*Ts)]

                        else:
                            
                            # alpha-function
                            b = [1-math.exp(-alpha*Ts)-(alpha*Ts)*math.exp(-alpha*Ts), math.exp(-2*alpha*Ts) + (alpha*Ts)*math.exp(-alpha*Ts) - math.exp(-alpha*Ts)]
                            a = [alpha**2, -2*(alpha**2)*math.exp(-alpha*Ts), (alpha**2)*math.exp(-2*alpha*Ts)]

                        # Calculate Signal without noise
                        signal_without_noise = signal.lfilter(b, a, impulse_series)
                        
                        # # Calculate Signal without noise (Foward-Backward)
                        # signal_without_noise = signal.filtfilt(b, a, impulse_series)
                        
                        final_signal_without_noise += signal_without_noise


                    # Plot Signal without noise
                    plt.figure(figsize=(8,7))
                    plt.plot(t, final_signal_without_noise)
                    plt.title('Signal without noise')
                    plt.xlabel('Time')
                    plt.ylabel('Signal Amplitude')
                    if math.isnan(alpha):
                        np.savetxt(f'{path_data}/signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.txt', signal_without_noise)
                        plt.savefig(f'{path_data}/signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.png', dpi = 600)
                    else:
                        np.savetxt(f'{path_data}/signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.txt', signal_without_noise)
                        plt.savefig(f'{path_data}/signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.png', dpi = 600)       
                    plt.close()

                    # Spectrum of signal

                    # Calculate fft of signal
                    signal_without_noise_norm = final_signal_without_noise - np.mean(final_signal_without_noise)

                    yf=np.fft.fft(signal_without_noise_norm)       
                     
                    # Calculate power spectrum
                    ps = abs(yf)**2

                    freqs = np.fft.fftfreq(len(t), Ts)
                    idx = np.argsort(freqs)

                    # Plot Power Spectrum    
                    plt.figure(figsize=(8,7))
                    plt.title('Power Spectrum of Signal without noise')
                    # Plot 1/2 spectrum
                    # plt.xscale('log')
                    # plt.yscale('log')
                    plt.xlabel('Frequencies')
                    plt.ylabel('Signal Power Spectrum')
                    plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                    # plt.xlim([0, 20])
                    # plt.ylim([0, 20])
                    if math.isnan(alpha):
                        np.savetxt(f'{path_data}/power_spectrum_signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                        plt.savefig(f'{path_data}/power_spectrum_signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.png', dpi = 600)
                    else:
                        np.savetxt(f'{path_data}/power_spectrum_signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                        plt.savefig(f'{path_data}/power_spectrum_signal_without_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.png', dpi = 600)       
                    plt.close()

                    plt.figure(figsize=(8,7))
                    plt.title('Spectrogram of Signal without noise')
                    f_spectro, t_spectro, Sxx = signal.spectrogram(signal_without_noise_norm, 1/Ts)
                    plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.close()



                    ## Simulation of Signal Noise
                    signal_variance = np.var(final_signal_without_noise)

                    for perc_sigma2 in perc_sigma2_list:
                        
                        print(f'noise variance percentual = {perc_sigma2}')                
                            
                        sigma2 = perc_sigma2*signal_variance
                        
                        np.random.seed(1)

                        noise = np.random.normal(average_noise, np.sqrt(sigma2), len(t))

                        # Plot Noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, noise)
                        plt.title('Signal Noise')
                        plt.xlabel('Time')
                        plt.ylabel('Noise Amplitude')
                        np.savetxt(f'{path_data}/noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', noise)
                        plt.savefig(f'{path_data}/noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        plt.close()




                        ## Signal with noise
                        signal_with_noise = final_signal_without_noise + noise

                        # noisy_signal_df = pd.DataFrame(np.insert(signal_with_noise, 0, 0))
                        noisy_signal_df = pd.DataFrame(signal_with_noise, columns=['0'])
                        
                        if math.isnan(alpha):
                            noisy_signal_df.to_csv(f'{path_data}/signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                        else:
                            noisy_signal_df.to_csv(f'{path_data}/signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)   
                    
                        
                        # Plot Signal without noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, signal_with_noise)
                        plt.title('Signal with noise')
                        plt.xlabel('Time')
                        plt.ylabel('Signal Amplitude')
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise)
                            plt.savefig(f'{path_data}/signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise)
                            plt.savefig(f'{path_data}/signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                        plt.close()


                        # Spectrum of signal

                        # Calculate fft of noisy signal
                        noisy_signal_norm = signal_with_noise - np.mean(signal_with_noise)

                        yf=np.fft.fft(noisy_signal_norm)

                        # Calculate power spectrum
                        ps = abs(yf)**2

                        freqs = np.fft.fftfreq(len(t), Ts)
                        idx = np.argsort(freqs)

                        # Plot Power Spectrum    
                        plt.figure(figsize=(8,7))
                        plt.title('Power Spectrum of Signal with noise')
                        # Plot 1/2 spectrum
                        # plt.xscale('log')
                        # plt.yscale('log')
                        plt.xlabel('Frequencies')
                        plt.ylabel('Signal Power Spectrum')
                        plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                        # plt.xlim([0, 20])
                        # plt.ylim([0, 20])
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                        plt.close()

                        plt.figure(figsize=(8,7))
                        plt.title('Spectrogram of Signal with noise')
                        f_spectro, t_spectro, Sxx = signal.spectrogram(noisy_signal_norm, 1/Ts)
                        plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                        plt.close()

                        ## Signal with noise integrated

                        signal_with_noise_int = np.cumsum(signal_with_noise - np.mean(signal_with_noise))

                        # noisy_signal_int_df = pd.DataFrame(np.insert(signal_with_noise_int, 0, 0))
                        noisy_signal_int_df = pd.DataFrame(signal_with_noise_int, columns=['0'])
                        
                        if math.isnan(alpha):
                            noisy_signal_int_df.to_csv(f'{path_data}/signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                        else:
                            noisy_signal_int_df.to_csv(f'{path_data}/signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)   
                    
                        # Plot Signal without noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, signal_with_noise_int)
                        plt.title('Signal with noise integrated')
                        plt.xlabel('Time')
                        plt.ylabel('Signal Amplitude')
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise_int)
                            plt.savefig(f'{path_data}/signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise_int)
                            plt.savefig(f'{path_data}/signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                        plt.close()

                        # Spectrum of signal

                        # Calculate fft of noisy signal integrated
                        noisy_signal_int_norm = signal_with_noise_int - np.mean(signal_with_noise_int)

                        yf=np.fft.fft(noisy_signal_int_norm)

                        # Calculate power spectrum
                        ps = abs(yf)**2

                        freqs = np.fft.fftfreq(len(t), Ts)
                        idx = np.argsort(freqs)

                        # Plot Power Spectrum    
                        plt.figure(figsize=(8,7))
                        plt.title('Power Spectrum of Signal with noise integrated')
                        # Plot 1/2 spectrum
                        # plt.xscale('log')
                        plt.yscale('log')
                        plt.xlabel('Frequencies')
                        plt.ylabel('Signal Power Spectrum')
                        plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                        # plt.xlim([0, 20])
                        # plt.ylim([0, 20])
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_int_r_p_{r_p}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                        plt.close()

                        plt.figure(figsize=(8,7))
                        plt.title('Spectrogram of Signal with noise integrated')
                        f_spectro, t_spectro, Sxx = signal.spectrogram(noisy_signal_int_norm, 1/Ts)
                        plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                        plt.close()
                
            elif WTs_distribution == 1:
                
                print(f'Start simulation Power-Law WTs distribution, seed {seed_value}')
                
                # Create path for saves
                if os.path.isdir(f'{mother_path}/output/Power_Law_WTs') == False:
                    directory = 'Power_Law_WTs'
                    parent_dir = f'{mother_path}/output'
                    path = os.path.join(parent_dir, directory) 
                    os.makedirs(path, exist_ok= True)
                    
                path_data = f'{mother_path}/output/Power_Law_WTs'
                
                for mu in mu_list:
                    
                    # Impulse Timing
                    impulse_WTs = []

                    if mu > 2:
                        T = tau*(mu - 2)
                    else:
                        T = tau

                    for i in range(0, n_impulse+1):
                        
                        if mu > 2:
                            impulse_WTs.append(T*(np.random.uniform(0,1,size = 1)**(1/(1-mu)) - 1))
                        else:
                            while True:
                                new_WT = T*(np.random.uniform(0,1,size = 1)**(1/(1-mu)) - 1)
                                
                                if new_WT <= 100:
                                    impulse_WTs.append(new_WT)
                                    break
                            
                    impulse_WTs = np.concatenate(impulse_WTs)

                    plt.figure(figsize=(8,7))
                    occurrences, wt = np.histogram(impulse_WTs, bins = int(max(impulse_WTs)))
                    plt.plot(wt[:-1], occurrences, '+')
                    plt.xscale('log')
                    plt.yscale('log')
                    # plt.xlim([0, 500])
                    plt.xlabel('WT')
                    plt.ylabel('# of occurrences')
                    plt.title('Histogram of WT')
                    np.savetxt(f'{path_data}/histogram_WTs_T_{round(T,3)}_mu_{mu}_seed_{seed_value}.txt', [wt[:-1], occurrences])
                    plt.savefig(f'{path_data}/histogram_WTs_T_{round(T,3)}_mu_{mu}_seed_{seed_value}.png', dpi = 600)
                    plt.close()

                    impulse_timing_full = np.cumsum(impulse_WTs)

                    impulse_timing = impulse_timing_full[:-1]

                    # Define sampling grid
                    t_max = max(impulse_timing_full)

                    t = np.arange(0, t_max, step = Ts)

                    np.savetxt(f'{path_data}/sampling_grid_seed_{seed_value}.txt', t)

                    # Impulse Amplitude
                    impulse_amplitudes = np.random.uniform(amp_min, amp_max, n_impulse)
                    
                    # Impulse Series
                    impulse_series = np.zeros(len(t))

                    # Find the indices in t that are closest to the impulse timings
                    # indices = np.abs(np.subtract.outer(t, impulse_timing)).argmin(axis=0)
                    indices = []
                    for temp in impulse_timing:
                        indices.append(np.abs(t - temp).argmin())

                    # Assign the impulse amplitudes at these indices
                    impulse_series[indices] = impulse_amplitudes
                        
                    # Plot Impulse series
                    plt.figure(figsize=(8,7))
                    plt.stem(t, impulse_series, markerfmt= ' ')
                    plt.title('Impulse Series')
                    plt.xlabel('Time')
                    plt.ylabel('Impulse Amplitude')
                    np.savetxt(f'{path_data}/impulse_series_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_seed_{seed_value}.txt', impulse_series)
                    plt.savefig(f'{path_data}/impulse_series_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_seed_{seed_value}.png', dpi = 600)
                    plt.close()

                    for omega_0_min, omega_0_max in zip(omega_0_min_list, omega_0_max_list):
                        
                        print(f'omega_0 range = [{omega_0_min}, {omega_0_max}]')    
                        
                        final_signal_without_noise = np.zeros(len(t))
                        
                        for omega_0 in np.arange(omega_0_min, omega_0_max, 1):
                            
                            if omega_0 == omega_0_min + (omega_0_max - omega_0_min)//4:
                                print('25%')
                                
                            elif omega_0 == omega_0_min + (omega_0_max - omega_0_min)//2:
                                
                                print('50%')
                                
                            elif omega_0 == omega_0_min + ((omega_0_max - omega_0_min)//4 + (omega_0_max - omega_0_min)//2):
                                
                                print('75%')
                                
                            ## Convolution of Impulse series with the Impulse response
                            if math.isnan(alpha):
                                
                                # Underdumped oscillator
                                omega_1 = (1/2)*np.sqrt(4*(omega_0**2) - (gamma**2))
                                cost = -2/(4*(omega_1**3) + (omega_1*(gamma**2)))                    
                                b = [cost*(math.exp(-(gamma/2)*Ts)*omega_1*2*math.cos(omega_1*Ts) + math.exp(-(gamma/2)*Ts)*gamma*math.sin(omega_1*Ts) - 2*omega_1), cost*(-math.exp(-gamma*Ts)*2*omega_1 + math.exp(-(gamma/2)*Ts)*omega_1*2*math.cos(omega_1*Ts) - math.exp(-(gamma/2)*Ts)*gamma*math.sin(omega_1*Ts))]
                                a = [1, -2*math.exp(-(gamma/2)*Ts)*math.cos(omega_1*Ts), math.exp(-gamma*Ts)]

                            else:
                                
                                # alpha-function
                                b = [1-math.exp(-alpha*Ts)-(alpha*Ts)*math.exp(-alpha*Ts), math.exp(-2*alpha*Ts) + (alpha*Ts)*math.exp(-alpha*Ts) - math.exp(-alpha*Ts)]
                                a = [alpha**2, -2*(alpha**2)*math.exp(-alpha*Ts), (alpha**2)*math.exp(-2*alpha*Ts)]

                            # Calculate Signal without noise
                            signal_without_noise = signal.lfilter(b, a, impulse_series)
                            
                            # # Calculate Signal without noise (Foward-Backward)
                            # signal_without_noise = signal.filtfilt(b, a, impulse_series)
                            
                            final_signal_without_noise += signal_without_noise


                        # Plot Signal without noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, final_signal_without_noise)
                        plt.title('Signal without noise')
                        plt.xlabel('Time')
                        plt.ylabel('Signal Amplitude')
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.txt', signal_without_noise)
                            plt.savefig(f'{path_data}/signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.txt', signal_without_noise)
                            plt.savefig(f'{path_data}/signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.png', dpi = 600)       
                        plt.close()

                        # Spectrum of signal

                        # Calculate fft of signal
                        signal_without_noise_norm = final_signal_without_noise - np.mean(final_signal_without_noise)

                        yf=np.fft.fft(signal_without_noise_norm)       
                         
                        # Calculate power spectrum
                        ps = abs(yf)**2

                        freqs = np.fft.fftfreq(len(t), Ts)
                        idx = np.argsort(freqs)

                        # Plot Power Spectrum    
                        plt.figure(figsize=(8,7))
                        plt.title('Power Spectrum of Signal without noise')
                        # Plot 1/2 spectrum
                        # plt.xscale('log')
                        # plt.yscale('log')
                        plt.xlabel('Frequencies')
                        plt.ylabel('Signal Power Spectrum')
                        plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                        # plt.xlim([0, 20])
                        # plt.ylim([0, 20])
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/power_spectrum_signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/power_spectrum_signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_without_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.png', dpi = 600)       
                        plt.close()

                        plt.figure(figsize=(8,7))
                        plt.title('Spectrogram of Signal without noise')
                        f_spectro, t_spectro, Sxx = signal.spectrogram(signal_without_noise_norm, 1/Ts)
                        plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                        plt.close()



                        ## Simulation of Signal Noise
                        signal_variance = np.var(final_signal_without_noise)

                        for perc_sigma2 in perc_sigma2_list:
                            
                            print(f'noise variance percentual = {perc_sigma2}')                
                                
                            sigma2 = perc_sigma2*signal_variance
                            
                            np.random.seed(1)

                            noise = np.random.normal(average_noise, np.sqrt(sigma2), len(t))

                            # Plot Noise
                            plt.figure(figsize=(8,7))
                            plt.plot(t, noise)
                            plt.title('Signal Noise')
                            plt.xlabel('Time')
                            plt.ylabel('Noise Amplitude')
                            np.savetxt(f'{path_data}/noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', noise)
                            plt.savefig(f'{path_data}/noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                            plt.close()

                            ## Signal with noise
                            signal_with_noise = final_signal_without_noise + noise

                            # noisy_signal_df = pd.DataFrame(np.insert(signal_with_noise, 0, 0))
                            noisy_signal_df = pd.DataFrame(signal_with_noise, columns=['0'])
                            
                            if math.isnan(alpha):
                                noisy_signal_df.to_csv(f'{path_data}/signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                            else:
                                noisy_signal_df.to_csv(f'{path_data}/signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)    
                        
                            # Plot Signal without noise
                            plt.figure(figsize=(8,7))
                            plt.plot(t, signal_with_noise)
                            plt.title('Signal with noise')
                            plt.xlabel('Time')
                            plt.ylabel('Signal Amplitude')
                            if math.isnan(alpha):
                                np.savetxt(f'{path_data}/signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise)
                                plt.savefig(f'{path_data}/signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                            else:
                                np.savetxt(f'{path_data}/signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise)
                                plt.savefig(f'{path_data}/signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                            plt.close()

                            # Spectrum of signal

                            # Calculate fft of noisy signal
                            noisy_signal_norm = signal_with_noise - np.mean(signal_with_noise)

                            yf=np.fft.fft(noisy_signal_norm)

                            # Calculate power spectrum
                            ps = abs(yf)**2

                            freqs = np.fft.fftfreq(len(t), Ts)
                            idx = np.argsort(freqs)

                            # Plot Power Spectrum    
                            plt.figure(figsize=(8,7))
                            plt.title('Power Spectrum of Signal with noise')
                            # Plot 1/2 spectrum
                            # plt.xscale('log')
                            # plt.yscale('log')
                            plt.xlabel('Frequencies')
                            plt.ylabel('Signal Power Spectrum')
                            plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                            # plt.xlim([0, 20])
                            # plt.ylim([0, 20])
                            if math.isnan(alpha):
                                np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                                plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                            else:
                                np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                                plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                            plt.close()

                            plt.figure(figsize=(8,7))
                            plt.title('Spectrogram of Signal with noise')
                            f_spectro, t_spectro, Sxx = signal.spectrogram(noisy_signal_norm, 1/Ts)
                            plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                            plt.ylabel('Frequency [Hz]')
                            plt.xlabel('Time [sec]')
                            plt.close()

                            ## Signal with noise integrated

                            signal_with_noise_int = np.cumsum(signal_with_noise - np.mean(signal_with_noise))

                            # noisy_signal_int_df = pd.DataFrame(np.insert(signal_with_noise_int, 0, 0))
                            noisy_signal_int_df = pd.DataFrame(signal_with_noise_int, columns=['0'])
                            
                            if math.isnan(alpha):
                                noisy_signal_int_df.to_csv(f'{path_data}/signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                            else:
                                noisy_signal_int_df.to_csv(f'{path_data}/signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)    
                                                
                            # Plot Signal without noise
                            plt.figure(figsize=(8,7))
                            plt.plot(t, signal_with_noise_int)
                            plt.title('Signal with noise integrated')
                            plt.xlabel('Time')
                            plt.ylabel('Signal Amplitude')
                            if math.isnan(alpha):
                                np.savetxt(f'{path_data}/signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise_int)
                                plt.savefig(f'{path_data}/signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                            else:
                                np.savetxt(f'{path_data}/signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise_int)
                                plt.savefig(f'{path_data}/signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                            plt.close()

                            # Spectrum of signal

                            # Calculate fft of noisy signal integrated
                            noisy_signal_int_norm = signal_with_noise_int - np.mean(signal_with_noise_int)

                            yf=np.fft.fft(noisy_signal_int_norm)

                            # Calculate power spectrum
                            ps = abs(yf)**2

                            freqs = np.fft.fftfreq(len(t), Ts)
                            idx = np.argsort(freqs)

                            # Plot Power Spectrum    
                            plt.figure(figsize=(8,7))
                            plt.title('Power Spectrum of Signal with noise integrated')
                            # Plot 1/2 spectrum
                            # plt.xscale('log')
                            plt.yscale('log')
                            plt.xlabel('Frequencies')
                            plt.ylabel('Signal Power Spectrum')
                            plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                            # plt.xlim([0, 20])
                            # plt.ylim([0, 20])
                            if math.isnan(alpha):
                                np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                                plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                            else:
                                np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                                plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_int_T_{round(T,3)}_mu_{mu}_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)       
                            plt.close()

                            plt.figure(figsize=(8,7))
                            plt.title('Spectrogram of Signal with noise integrated')
                            f_spectro, t_spectro, Sxx = signal.spectrogram(noisy_signal_int_norm, 1/Ts)
                            plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                            plt.ylabel('Frequency [Hz]')
                            plt.xlabel('Time [sec]')
                            plt.close()
                    
            else:
                
                print(f'Start simulation Uniform WTs distribution, seed {seed_value}')  
                
                # Create path for saves
                if os.path.isdir(f'{mother_path}/output/Uniform_WTs') == False:
                    directory = 'Uniform_WTs'
                    parent_dir = f'{mother_path}/output'
                    path = os.path.join(parent_dir, directory) 
                    os.makedirs(path, exist_ok= True)
                    
                path_data = f'{mother_path}/output/Uniform_WTs' 
                
                # Impulse Timing
                impulse_WTs = np.random.uniform(0, 2, n_impulse +1)

                plt.figure(figsize=(8,7))
                occurrences, wt = np.histogram(impulse_WTs, bins = 50)
                plt.plot(wt[:-1], occurrences, '+')
                # plt.xscale('log')
                # plt.yscale('log')
                # plt.xlim([0, 500])
                plt.xlabel('WT')
                plt.ylabel('# of occurrences')
                plt.title('Histogram of WT')
                np.savetxt(f'{path_data}/histogram_WTs_seed_{seed_value}.txt', [wt[:-1], occurrences])
                plt.savefig(f'{path_data}/histogram_WTs_seed_{seed_value}.png', dpi = 600)
                plt.close()

                impulse_timing_full = np.cumsum(impulse_WTs)

                impulse_timing = impulse_timing_full[:-1]

                # Define sampling grid
                t_max = max(impulse_timing_full)

                t = np.arange(0, t_max, step = Ts)

                np.savetxt(f'{path_data}/sampling_grid_seed_{seed_value}.txt', t)

                # Impulse Amplitude
                impulse_amplitudes = np.random.uniform(amp_min, amp_max, n_impulse)
                
                # Impulse Series
                impulse_series = np.zeros(len(t))

                # Find the indices in t that are closest to the impulse timings
                # indices = np.abs(np.subtract.outer(t, impulse_timing)).argmin(axis=0)
                indices = []
                for temp in impulse_timing:
                    indices.append(np.abs(t - temp).argmin())

                # Assign the impulse amplitudes at these indices
                impulse_series[indices] = impulse_amplitudes
                    
                # Plot Impulse series
                plt.figure(figsize=(8,7))
                plt.stem(t, impulse_series, markerfmt= ' ')
                plt.title('Impulse Series')
                plt.xlabel('Time')
                plt.ylabel('Impulse Amplitude')
                np.savetxt(f'{path_data}/impulse_series_n_impulses_{n_impulse}_seed_{seed_value}.txt', impulse_series)
                plt.savefig(f'{path_data}/impulse_series_n_impulses_{n_impulse}_seed_{seed_value}.png', dpi = 600)
                plt.close()

                for omega_0_min, omega_0_max in zip(omega_0_min_list, omega_0_max_list):
                    
                    print(f'omega_0 range = [{omega_0_min}, {omega_0_max}]')    
                    
                    final_signal_without_noise = np.zeros(len(t))
                    
                    for omega_0 in np.arange(omega_0_min, omega_0_max, 1):
                        
                        if omega_0 == omega_0_min + (omega_0_max - omega_0_min)//4:
                            print('25%')
                            
                        elif omega_0 == omega_0_min + (omega_0_max - omega_0_min)//2:
                            
                            print('50%')
                            
                        elif omega_0 == omega_0_min + ((omega_0_max - omega_0_min)//4 + (omega_0_max - omega_0_min)//2):
                            
                            print('75%')
                            
                        ## Convolution of Impulse series with the Impulse response
                        if math.isnan(alpha):
                            
                            # Underdumped oscillator
                            omega_1 = (1/2)*np.sqrt(4*(omega_0**2) - (gamma**2))
                            cost = -2/(4*(omega_1**3) + (omega_1*(gamma**2)))                    
                            b = [cost*(math.exp(-(gamma/2)*Ts)*omega_1*2*math.cos(omega_1*Ts) + math.exp(-(gamma/2)*Ts)*gamma*math.sin(omega_1*Ts) - 2*omega_1), cost*(-math.exp(-gamma*Ts)*2*omega_1 + math.exp(-(gamma/2)*Ts)*omega_1*2*math.cos(omega_1*Ts) - math.exp(-(gamma/2)*Ts)*gamma*math.sin(omega_1*Ts))]
                            a = [1, -2*math.exp(-(gamma/2)*Ts)*math.cos(omega_1*Ts), math.exp(-gamma*Ts)]

                        else:
                            
                            # alpha-function
                            b = [1-math.exp(-alpha*Ts)-(alpha*Ts)*math.exp(-alpha*Ts), math.exp(-2*alpha*Ts) + (alpha*Ts)*math.exp(-alpha*Ts) - math.exp(-alpha*Ts)]
                            a = [alpha**2, -2*(alpha**2)*math.exp(-alpha*Ts), (alpha**2)*math.exp(-2*alpha*Ts)]

                        # Calculate Signal without noise
                        signal_without_noise = signal.lfilter(b, a, impulse_series)
                        
                        # # Calculate Signal without noise (Foward-Backward)
                        # signal_without_noise = signal.filtfilt(b, a, impulse_series)
                        
                        final_signal_without_noise += signal_without_noise


                    # Plot Signal without noise
                    plt.figure(figsize=(8,7))
                    plt.plot(t, final_signal_without_noise)
                    plt.title('Signal without noise')
                    plt.xlabel('Time')
                    plt.ylabel('Signal Amplitude')
                    if math.isnan(alpha):
                        np.savetxt(f'{path_data}/signal_without_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.txt', signal_without_noise)
                        plt.savefig(f'{path_data}/signal_without_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.png', dpi = 600)
                    else:
                        np.savetxt(f'{path_data}/signal_without_noise_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.txt', signal_without_noise)
                        plt.savefig(f'{path_data}/signal_without_noise_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.png', dpi = 600)      
                    plt.close()

                    # Spectrum of signal

                    # Calculate fft of signal
                    signal_without_noise_norm = final_signal_without_noise - np.mean(final_signal_without_noise)

                    yf=np.fft.fft(signal_without_noise_norm)       
                     
                    # Calculate power spectrum
                    ps = abs(yf)**2

                    freqs = np.fft.fftfreq(len(t), Ts)
                    idx = np.argsort(freqs)

                    # Plot Power Spectrum    
                    plt.figure(figsize=(8,7))
                    plt.title('Power Spectrum of Signal without noise')
                    # Plot 1/2 spectrum
                    # plt.xscale('log')
                    # plt.yscale('log')
                    plt.xlabel('Frequencies')
                    plt.ylabel('Signal Power Spectrum')
                    plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                    # plt.xlim([0, 20])
                    # plt.ylim([0, 20])
                    if math.isnan(alpha):
                        np.savetxt(f'{path_data}/power_spectrum_signal_without_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                        plt.savefig(f'{path_data}/power_spectrum_signal_without_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_seed_{seed_value}.png', dpi = 600)
                    else:
                        np.savetxt(f'{path_data}/power_spectrum_signal_without_noise_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                        plt.savefig(f'{path_data}/power_spectrum_signal_without_noise_n_impulses_{n_impulse}_alpha_{alpha}_seed_{seed_value}.png', dpi = 600) 
                    plt.close()

                    plt.figure(figsize=(8,7))
                    plt.title('Spectrogram of Signal without noise')
                    f_spectro, t_spectro, Sxx = signal.spectrogram(signal_without_noise_norm, 1/Ts)
                    plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.close()



                    ## Simulation of Signal Noise
                    signal_variance = np.var(final_signal_without_noise)

                    for perc_sigma2 in perc_sigma2_list:
                        
                        print(f'noise variance percentual = {perc_sigma2}')                
                            
                        sigma2 = perc_sigma2*signal_variance
                        
                        np.random.seed(1)

                        noise = np.random.normal(average_noise, np.sqrt(sigma2), len(t))

                        # Plot Noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, noise)
                        plt.title('Signal Noise')
                        plt.xlabel('Time')
                        plt.ylabel('Noise Amplitude')
                        np.savetxt(f'{path_data}/noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', noise)
                        plt.savefig(f'{path_data}/noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        plt.close()

                        ## Signal with noise
                        signal_with_noise = final_signal_without_noise + noise

                        # noisy_signal_df = pd.DataFrame(np.insert(signal_with_noise, 0, 0))
                        noisy_signal_df = pd.DataFrame(signal_with_noise, columns=['0'])

                        if math.isnan(alpha):
                            noisy_signal_df.to_csv(f'{path_data}/signal_with_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                        else:
                            noisy_signal_df.to_csv(f'{path_data}/signal_with_noise_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)                            

                        # Plot Signal without noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, signal_with_noise)
                        plt.title('Signal with noise')
                        plt.xlabel('Time')
                        plt.ylabel('Signal Amplitude')
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/signal_with_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise)
                            plt.savefig(f'{path_data}/signal_with_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/signal_with_noise_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise)
                            plt.savefig(f'{path_data}/signal_with_noise_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600) 
                        plt.close()

                        # Spectrum of signal

                        # Calculate fft of noisy signal
                        noisy_signal_norm = signal_with_noise - np.mean(signal_with_noise)

                        yf=np.fft.fft(noisy_signal_norm)

                        # Calculate power spectrum
                        ps = abs(yf)**2

                        freqs = np.fft.fftfreq(len(t), Ts)
                        idx = np.argsort(freqs)

                        # Plot Power Spectrum    
                        plt.figure(figsize=(8,7))
                        plt.title('Power Spectrum of Signal with noise')
                        # Plot 1/2 spectrum
                        # plt.xscale('log')
                        # plt.yscale('log')
                        plt.xlabel('Frequencies')
                        plt.ylabel('Signal Power Spectrum')
                        plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                        # plt.xlim([0, 20])
                        # plt.ylim([0, 20])
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600) 
                        plt.close()

                        plt.figure(figsize=(8,7))
                        plt.title('Spectrogram of Signal with noise')
                        f_spectro, t_spectro, Sxx = signal.spectrogram(noisy_signal_norm, 1/Ts)
                        plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                        plt.close()

                        ## Signal with noise integrated

                        signal_with_noise_int = np.cumsum(signal_with_noise - np.mean(signal_with_noise))

                        # noisy_signal_int_df = pd.DataFrame(np.insert(signal_with_noise_int, 0, 0))
                        noisy_signal_int_df = pd.DataFrame(signal_with_noise_int, columns=['0'])

                        if math.isnan(alpha):
                            noisy_signal_int_df.to_csv(f'{path_data}/signal_with_noise_int_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                        else:
                            noisy_signal_int_df.to_csv(f'{path_data}/signal_with_noise_int_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.csv', index = False, header = True)
                            
                        # Plot Signal without noise
                        plt.figure(figsize=(8,7))
                        plt.plot(t, signal_with_noise_int)
                        plt.title('Signal with noise integrated')
                        plt.xlabel('Time')
                        plt.ylabel('Signal Amplitude')
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/signal_with_noise_int_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise_int)
                            plt.savefig(f'{path_data}/signal_with_noise_int_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/signal_with_noise_int_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', signal_with_noise_int)
                            plt.savefig(f'{path_data}/signal_with_noise_int_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600) 
                        plt.close()

                        # Spectrum of signal

                        # Calculate fft of noisy signal integrated
                        noisy_signal_int_norm = signal_with_noise_int - np.mean(signal_with_noise_int)

                        yf=np.fft.fft(noisy_signal_int_norm)

                        # Calculate power spectrum
                        ps = abs(yf)**2

                        freqs = np.fft.fftfreq(len(t), Ts)
                        idx = np.argsort(freqs)

                        # Plot Power Spectrum    
                        plt.figure(figsize=(8,7))
                        plt.title('Power Spectrum of Signal with noise integrated')
                        # Plot 1/2 spectrum
                        # plt.xscale('log')
                        plt.yscale('log')
                        plt.xlabel('Frequencies')
                        plt.ylabel('Signal Power Spectrum')
                        plt.plot(freqs[idx[len(idx)//2+1:]], ps[idx[len(idx)//2+1:]])
                        # plt.xlim([0, 20])
                        # plt.ylim([0, 20])
                        if math.isnan(alpha):
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_int_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_int_n_impulses_{n_impulse}_omega0_{omega_0_min}_{omega_0_max}_gamma_{gamma}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600)
                        else:
                            np.savetxt(f'{path_data}/power_spectrum_signal_with_noise_int_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.txt', [freqs[idx], ps[idx]])
                            plt.savefig(f'{path_data}/power_spectrum_signal_with_noise_int_n_impulses_{n_impulse}_alpha_{alpha}_noise_mean_{average_noise}_perc_var_{perc_sigma2}_variance_{format(sigma2, ".2e")}_seed_{seed_value}.png', dpi = 600) 
                        plt.close()

                        plt.figure(figsize=(8,7))
                        plt.title('Spectrogram of Signal with noise integrated')
                        f_spectro, t_spectro, Sxx = signal.spectrogram(noisy_signal_int_norm, 1/Ts)
                        plt.pcolormesh(t_spectro, f_spectro, Sxx, shading='gouraud')
                        plt.ylabel('Frequency [Hz]')
                        plt.xlabel('Time [sec]')
                        plt.close()

print("--- %s seconds for eeg_signal_simulator_frequency_bands.py ---" % (tm.time() - start_time_code))
