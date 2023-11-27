#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from scipy.signal import find_peaks
#from PyEMD import CEEMDAN
from scipy.fft import fft, fftfreq
from scipy.signal import blackman
from scipy.signal import find_peaks
import statsmodels.api as sm
from scipy.signal import butter, lfilter,filtfilt,sosfilt
from scipy.stats import kurtosis,skew
directory = os.getcwd()
if directory[0] == "/":
    linux = True
else:
    linux = False
#%% Import scripts
import sys
if linux == True:  
    directory_functions = str(directory +"/functions/")
    sys.path.insert(0, directory_functions) # linux
else:
    directory_functions = str(directory +"\\functions\\")
    sys.path.insert(0, directory_functions) # linux

import functions_paths as fpt

#%% Functions 
def load_dataset_activations(dataset_name,linux:bool=False):
    
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + dataset_name + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + dataset_name + ".csv" # Windows
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    
    df = df
    
    return df

def band_pass_filter(signal,high:float=.01,low:float=.15,order:int=3,fs:float=0,verbose:bool=True):
    if fs == 0:
        sos = butter(order, high,'high', output='sos')
        filtered = sosfilt(sos, signal)
        if verbose==True:
            plt.plot(filtered)
            plt.show()
        sos = butter(order, low, 'low', output='sos')
        filtered = sosfilt(sos, filtered)
    else:
        b, a = butter(order, high,'high',fs=fs)
        filtered = filtfilt(b, a, signal)
        if verbose==True:
            plt.plot(filtered)
            plt.show()
        b, a = butter(order, low,'low',fs=fs)
        filtered = filtfilt(b, a, filtered)
        
    if verbose==True:
        plt.plot(filtered)
        plt.show()
        
    return filtered

def decompose_signal(signal,high,low,fs,n_levels:int=10,show_plot:bool=True):
    signals_mat = np.zeros((n_levels,int(len(signal))))
    step = (low-high)/n_levels
    for i in range(n_levels):    
        signal_i = band_pass_filter(signal,high=low-step,low=low,fs=fs,verbose=False)
        signals_mat[i,:] = signal_i
        low -= step    
    
        if show_plot==True:
            plt.show()
            plt.plot(signal_i)           
    
    return signals_mat

def initialize_CEEMDAN(trials_f:int=100,epsilon_f:float=0.005,parallel_f:bool=False,processes_f:int=1,noise_scale_f:float=1.0):
    ceemdan = CEEMDAN(trials=trials_f, epsilon=epsilon_f, processes=processes_f)
    ceemdan.noise_seed(0)    
    return ceemdan


def dominant_frequencies_imfs(imf_mat,show_plot:bool=False,windowing:bool=True,point_seconds:float=1):
    total_imfs = len(imf_mat)        
    dominant_frequencies_vec = np.zeros(total_imfs)    
        
    for i in range(0,total_imfs):
        current_imf = imf_mat[i,:]     
        no_frequencies = 1
        dominant_frequencies_vec[i] = dominant_frequency(current_imf,no_frequencies,show_plot,windowing)                
                
    return dominant_frequencies_vec    
    
        
def dominant_frequency(signal,no_freq:int=1,show_plot:bool=False,windowing:bool=True,point_seconds:float=1):        
    # Number of sample points        
    N = len(signal)
    # sample spacing
    T = point_seconds  # (sampling rate)     
    if windowing==True:        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
    else:
        w = 1
        
    signal_f = fft(signal*w)    
    freqs = fftfreq(N, T)[:N//2] 
    positive_real_fs = np.abs(signal_f[0:N//2]) # only positive real values
    freq_step = freqs[1]
   
    # plot positive real half of the spectrum 
    if show_plot == True:
        #plot n values of the spectrum
        n = 50
        plt.plot(freqs[range(0,n)],  positive_real_fs[range(0,n)])    
        if windowing == True:
            plt.title(label="Spectrum " + "with blackman window")
        else:
            plt.title(label="Spectrum " + "without blackman window")
        plt.grid()    
        plt.show()
    
    sorted_indices = np.argsort(positive_real_fs)
    
    ans = np.zeros(no_freq)
    cont = 0
    for i in range(1,no_freq+1):        
        ans[cont] = sorted_indices[(i * -1)] * freq_step
        cont = cont + 1        

    return ans

 
def autocorrelation_imfs(imf_mat,verbosity:bool=False):
    no_imfs = np.shape(imf_mat)[0]
    ac_vec = []
    ac_mean_vec = []
    for i in range(no_imfs):
        #calculate autocorrelation
        ac_imf = sm.tsa.acf(imf_mat[i,:],nlags=int(np.ceil(len(imf_mat[i,:])/10)),fft=False)
        ac_vec.append(ac_imf)
        ac_mean = np.mean(ac_imf)
        ac_mean_vec.append(ac_mean)

    if verbosity == True:     
        print("\n")
        plt.bar(range(0,no_imfs),ac_mean_vec)        
        plt.title("mean auctocorrelation with nlags=len(signal)/10")
        plt.show()
        
    return ac_vec,ac_mean_vec


def concentration_energy_spectrum(imf_mat,no_imfs,verbosity:bool=True,title_plot:str="Energy concentration"):
    perc_max_vec = []
    for z in range(no_imfs):    
        signal = imf_mat[z,:]
        # Number of sample points
        N = len(signal)
        # sample spacing        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
        signal_f = fft(signal*w) # Multiplie the original signal with the window and get the fourier transform      
        positive_real_fs = np.abs(signal_f[0:N//2]) # Only positive real values        
        # Get the peaks from the signal 
        peaks_position,peak_heights = find_peaks(positive_real_fs)
        peaks_values = positive_real_fs[peaks_position]
        suma = 0
        try:
            maxi = np.max(peaks_values)
            for i in range(0,len(peaks_values)):
                if peaks_values[i] == maxi:
                    suma = suma
                else:
                    suma = suma + peaks_values[i]    
                    
            one_hundred = suma+maxi
            perc_maximum = (maxi*100)/one_hundred
            perc_max_vec.append(perc_maximum)
            if verbosity == True:
                print("percentage of the maximumn:",perc_maximum)
        except:
            maxi = 0
            perc_max_vec.append(maxi)
                      
    if verbosity == True:
        plt.bar(np.arange(0,len(perc_max_vec),1),perc_max_vec)
        plt.title(title_plot)        
        plt.xlabel('IMF')
        plt.ylabel('Percentage in dm freq')
        plt.show()        
        
    return perc_max_vec

def statistics_energy_spectrum(signal):
        # Number of sample points
        N = len(signal)
        # sample spacing        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
        signal_f = fft(signal*w) # Multiplie the original signal with the window and get the fourier transform      
        positive_real_fs = np.abs(signal_f[0:N//2]) # Only positive real values 
        var_s = np.var(positive_real_fs)
        mean_s = np.mean(positive_real_fs)
        skew_s = skew(positive_real_fs)
        kurt_s = kurtosis(positive_real_fs)
        return mean_s,var_s,skew_s,kurt_s

def variance_imfs(imf_mat,verbosity:bool=False):
    variance_vec = []
    no_imfs = np.shape(imf_mat)[0]
    
    for i in range(no_imfs):
        variance_vec.append(np.var(imf_mat[i,:]))
        if verbosity == True:
            print("Variance imf ", i ,": ",variance_vec[i])
            
    if verbosity == True:
        print("\n")   
        plt.bar(range(0,no_imfs),variance_vec)        
        plt.title("Variance in IMFS")
        plt.show()
        
    return variance_vec

def correlation_between_signals(signal,target_signal,verbosity:bool=False):
    stt = np.std(target_signal)
    sts = np.std(signal)
    if stt == 0.0 or sts == 0.0:
        print("\n Standar deviation in the signal is 0")
        return np.array([[0,0],[0,0]])
    corr = np.corrcoef(signal,target_signal)
    if verbosity == True:        
        print("\nCorrelation: \n",corr)
    return corr

def max_overlap_lags(signal,freq,n_lags,sampling_rate,no_points,show_plots:bool=False):
    # First generate a sine wave that is the baseline
    start_time = 0
    end_time = no_points*sampling_rate
    time = np.arange(start_time, no_points, 1)
    theta = 0
    step = (3.1416 * 2)/n_lags
    amplitude = 1
    corr_i = 0
    max_overlap = 0
    # Now look for the maximal correlation 
    for i in range(n_lags):
        theta += step
        sinewave = amplitude * np.sin(2 * np.pi * freq * time + theta)
        co = correlation_between_signals(signal, sinewave)[0,1]
        if show_plots==True:
            plt.show()
            plt.plot(sinewave)            
        if co >= corr_i:
            max_overlap = i
            corr_i = co
            
    return max_overlap


def max_overlap_mat(decomp_mat,frequencies,no_lags,sampling_rate):
    no_samples = np.shape(decomp_mat)[0]
    vec_z = np.zeros(no_samples)
    for i in range(no_samples):
        freq = frequencies[i]
        vec_z[i] = max_overlap_lags(decomp_mat[i,:],freq,no_lags,sampling_rate,np.shape(decomp_mat)[1],show_plots=False)
        
    return vec_z


def main_feature_extraction(dataset,no_rois,no_timepoints):
    show_plots = False
    percentage = 100
    no_features = 58
    no_columns = int(no_features * no_rois)
    no_rows = int(np.shape(dataset)[0]/no_rois)
    feature_mat = np.zeros((no_rows,no_columns))
    
    tic = time.perf_counter()
    cont = 0 
    for i in range(no_rows):
        
        feature_vector = []
        
        for j in range(no_rois):
            roi_patient = cont
            time_series = dataset[roi_patient,:no_timepoints]
            t = np.linspace(0, 1, len(time_series))
            
            # Clean the signal by filtering the frequencies under and above the interesting ones
            tr = tr_vec[roi_patient]
            fs = 1/tr
            low = .15 # Pass all the low frequencies under .15 hz, cut the rest
            high = .01 # Pass al the high frequencies over .01 hz, cut the rest
            
            # Obtain the signal filtered with 15 different band pass filters
            decomp_mat = decompose_signal(time_series,high,low,fs,n_levels=10,show_plot=False)
            
            # Filter the original signal to have only the relevant frequencies
            time_series = band_pass_filter(time_series,high=high,low=low,fs=fs,verbose=False)
                        
            # Features from the original Signal             
            # Statistical features from the main signal 
            feature_vector.append(np.mean(time_series))
            feature_vector.append(np.var(time_series))
            
            # Dominant Frequency of the main signal 
            feature_vector.append(dominant_frequency(time_series)[0])
            
            # Mean autocorrelation of the main signal 
            ac_imf = sm.tsa.acf(time_series,nlags=int(np.ceil(len(time_series)/10)),fft=False)
            feature_vector.append(np.mean(ac_imf))
            
            ## Statistics from the spectrogram skewnes and kurtosis
            see = statistics_energy_spectrum(time_series)
            feature_vector.append(see[2])
            feature_vector.append(see[3])
            
            # Dominant Frequencies from the IMFS and important info
            dominant_frequencies = dominant_frequencies_imfs(decomp_mat,percentage,show_plots,point_seconds=tr)     
            feature_vector.append(dominant_frequencies[0])
            feature_vector.append(dominant_frequencies[1])
            feature_vector.append(dominant_frequencies[2])
            feature_vector.append(dominant_frequencies[3])
            feature_vector.append(dominant_frequencies[4])
            feature_vector.append(dominant_frequencies[5])
            feature_vector.append(dominant_frequencies[6])
            feature_vector.append(dominant_frequencies[7])
            feature_vector.append(dominant_frequencies[8])
            feature_vector.append(dominant_frequencies[9])
            
            feature_vector.append(min(dominant_frequencies))
            feature_vector.append(max(dominant_frequencies))
            
            # Avg autocorrelations of the IMFS 
            avg_autocorrelations = autocorrelation_imfs(decomp_mat)[1]  
            feature_vector.append(avg_autocorrelations[0])
            feature_vector.append(avg_autocorrelations[1])
            feature_vector.append(avg_autocorrelations[2])
            feature_vector.append(avg_autocorrelations[3])
            feature_vector.append(avg_autocorrelations[4])
            feature_vector.append(avg_autocorrelations[5])
            feature_vector.append(avg_autocorrelations[6])
            feature_vector.append(avg_autocorrelations[7])
            feature_vector.append(avg_autocorrelations[8])
            feature_vector.append(avg_autocorrelations[9])
            
            # Concentration of energy of the IMFS   
            concentration_energy_imfs = concentration_energy_spectrum(decomp_mat, np.shape(decomp_mat)[0],verbosity=False)
            feature_vector.append(concentration_energy_imfs[0])
            feature_vector.append(concentration_energy_imfs[1])
            feature_vector.append(concentration_energy_imfs[2])
            feature_vector.append(concentration_energy_imfs[3])
            feature_vector.append(concentration_energy_imfs[4])
            feature_vector.append(concentration_energy_imfs[5])
            feature_vector.append(concentration_energy_imfs[6])
            feature_vector.append(concentration_energy_imfs[7])
            feature_vector.append(concentration_energy_imfs[8])
            feature_vector.append(concentration_energy_imfs[9])
            
            # Var of the IMFS
            var_imfs = variance_imfs(decomp_mat)
            feature_vector.append(var_imfs[0])
            feature_vector.append(var_imfs[1])
            feature_vector.append(var_imfs[2])
            feature_vector.append(var_imfs[3])
            feature_vector.append(var_imfs[4])
            feature_vector.append(var_imfs[5])
            feature_vector.append(var_imfs[6])
            feature_vector.append(var_imfs[7])
            feature_vector.append(var_imfs[8])
            feature_vector.append(var_imfs[9])
            
            # Global possition of the signal through different lags
            no_lags = 15
            frequencies_vec = [.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.11,.12,.13,.14,.15]
            max_overlap_vec = max_overlap_mat(decomp_mat,frequencies_vec,no_lags,tr)
            
            feature_vector.append(max_overlap_vec[0])
            feature_vector.append(max_overlap_vec[1])
            feature_vector.append(max_overlap_vec[2])
            feature_vector.append(max_overlap_vec[3])
            feature_vector.append(max_overlap_vec[4])
            feature_vector.append(max_overlap_vec[5])
            feature_vector.append(max_overlap_vec[6])
            feature_vector.append(max_overlap_vec[7])
            feature_vector.append(max_overlap_vec[8])
            feature_vector.append(max_overlap_vec[9])
            
            cont += 1
        
        feature_mat[i,:] = feature_vector
        
        perc = (i*100) / no_rows
        print(" " + str(perc) + "%")
        
    toc = time.perf_counter()
    print("\nTime elapsed: ",(toc-tic), "seconds")
    
    return feature_mat

#%% Dataset loading

dataset_name = "activations"
dataset = load_dataset_activations(dataset_name,linux=linux)

subjects_vec = dataset.Id
roi_vec = dataset.Roi
class_vec = dataset.Class
dataset_vec = dataset.Dataset
tr_vec = dataset.Tr
no_points_vec = dataset.No_points

dataset = dataset.drop('Id', axis=1)
dataset = dataset.drop('Roi', axis=1)
dataset = dataset.drop('Class', axis=1)
dataset = dataset.drop('No_points', axis=1)
dataset = dataset.drop('Tr',axis=1)
dataset = dataset.drop('Dataset', axis=1)

dataset = np.array(dataset)
#%% cleaning the signal with filters and CEEMDAN decomposition
no_rois = 200
no_timepoints = 175
feature_mat = main_feature_extraction(dataset,no_rois,no_timepoints)
df = pd.DataFrame(feature_mat)

subjects_vec_1 = []
class_vec_1 = np.zeros(np.shape(feature_mat)[0])
cont = 0
for i in range(np.shape(feature_mat)[0]):
    subjects_vec_1.append(subjects_vec[cont])
    class_vec_1[i] = class_vec[cont]
    cont += no_rois

df.insert(0, "Id", subjects_vec_1, True)
df.insert(0,"Class",class_vec_1,True)

feature_vector_names = ["mean_signal","var_signal","dominant_frequency_signal","mean_autocorr_signal","fft_skew","fft_kurtosis","dominant_freq_1","dominant_freq_2","dominant_freq_3","dominant_freq_4","dominant_freq_5","dominant_freq_6","dominant_freq_7","dominant_freq_8","dominant_freq_9","dominant_freq_10","max_dominant_freq","min_dominant_freq","avg_autocorr_1","avg_autocorr_2","avg_autocorr_3","avg_autocorr_4","avg_autocorr_5","avg_autocorr_6","avg_autocorr_7","avg_autocorr_8","avg_autocorr_9","avg_autocorr_10","conentration_enrgy_1","conentration_enrgy_2","conentration_enrgy_3","conentration_enrgy_4","conentration_enrgy_5","conentration_enrgy_6","conentration_enrgy_7","conentration_enrgy_8","conentration_enrgy_9","conentration_enrgy_10","var_1","var_2","var_3","var_4","var_5","var_6","var_7","var_8","var_9","var_10","max_overlap_lag_1","max_overlap_lag_2","max_overlap_lag_3","max_overlap_lag_4","max_overlap_lag_5","max_overlap_lag_6","max_overlap_lag_7","max_overlap_lag_8","max_overlap_lag_9","max_overlap_lag_10"]

feature_vector_names_complete = [None] * (no_rois * len(feature_vector_names) + 2)
feature_vector_names_complete[0] = "Class"
feature_vector_names_complete[1] = "Id"

cont = 2
for i in range(no_rois):
    for j in range(len(feature_vector_names)):
        feature_vector_names_complete[cont] = feature_vector_names[j] + "_roi" + str(i+1)
        cont += 1

df.columns = feature_vector_names_complete
df.to_csv("features.csv",index=False)













