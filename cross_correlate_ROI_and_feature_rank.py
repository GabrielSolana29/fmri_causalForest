# Cross-correlate the average of the feature relevance with the ROI relevance

#### Analysis of the data that we got from the computers 
## For the wrapper results
### id + number_ + features
## For the causal forest results
### id_c + number_ + causal_features  
## Numbers go from 1 to 4 
import pandas as pd
import numpy as np 
import copy
import matplotlib.pyplot as plt
import prince
from itertools import groupby

linux = False
import os
#%% Import scripts
#from xgboost import XGBRegressor
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

def remove_nan_target(list_features):
    new_list = []
    for i in range(len(list_features)):
        if type(list_features[i]) != float and list_features[i] != "target": 
            new_list.append(list_features[i])
            
    return new_list

def list_all_features_that_appear(mat):
    new_list = []
    
    for i in range(np.shape(mat)[0]):
        list_iter = remove_nan_target(mat[i,:])
        for j in range(len(list_iter)):
            if (list_iter[j] in new_list) == False:
                new_list.append(list_iter[j])
                
    return new_list


def list_all_iterations(mat):
    complete_list = []
    
    for i in range(np.shape(mat)[0]):
        list_iter = remove_nan_target(mat[i,:])
        for j in range(len(list_iter)):
            complete_list.append(list_iter[j])
            
    return complete_list

def number_of_appearances(mat,name_feat):
    number_app = 0    
    list_features = list_all_iterations(mat)
    
    for i in range(len(list_features)):
        if list_features[i] == name_feat:
            number_app += 1
            
    return number_app
    

def rank_feat(list_features,name_feature):
    for i in range(len(list_features)):
        if list_features[i] == name_feature:
            return i 
    ## If the feature was not selected in the list it will add a big number to its count    
    return len(list_features)+1
    
def ranking_hist(mat,name_feat):
    rank = []
    for i in range(np.shape(mat)[0]):
        list_iter = remove_nan_target(mat[i,:])
        list_iter = list_iter[::-1]
        r = rank_feat(list_iter,name_feat)
        rank.append(r)
            
    return rank

def get_all_info_feature(mat_o,name_feature):
    num_ap = number_of_appearances(mat_o,name_feature)

    rank_h = ranking_hist(mat_o,name_feature)
    avg_rank = np.average(rank_h)
    #statistical characteristics
    variance_feature = np.var(rank_h)
    mean_importance = np.average(rank_h)
    
    return name_feature,num_ap,rank_h,variance_feature,mean_importance

def frequency_appearance_avg_rank_variance(mat_o):
    all_features = list_all_features_that_appear(mat_o)
    feature_times_rank_variance = np.zeros((len(all_features),3))
    
    #See the number of aparition
    for i in range(np.shape(mat_o)[0]):
        list_iter = remove_nan_target(mat_o[i,:])
        for j in range(len(all_features)):
            if all_features[j] in list_iter:
                feature_times_rank_variance[j,0] += 1.0
                
    # Calculate the rank of the feature and the variance  
    for i in range(len(all_features)):    
        rank_h = ranking_hist(mat_o,all_features[i])
        avg_rank = np.average(rank_h)
        variance_feature = np.var(rank_h)
        feature_times_rank_variance[i,1] = avg_rank
        feature_times_rank_variance[i,2] = variance_feature
    
    return feature_times_rank_variance, all_features
    
def avg_variance_n_features(mat_o,num_features):

    ## see how many times does the feature appear in the iterations

    return
    
def sort_all_features(vec,all_features):
    imp_vec = vec[:,1]
    imp_vec_sort = np.argsort(imp_vec)
    new_features = copy.copy(all_features)
    new_vec = np.zeros((np.shape(vec)[0],np.shape(vec)[1]))
    for i in range(len(new_features)):
        index_i = int(imp_vec_sort[i])
        new_features[i] = all_features[index_i] 
        new_vec[i,0] = vec[index_i,0] 
        new_vec[i,1] = vec[index_i,1]
        new_vec[i,2] = vec[index_i,2]
    
    return new_vec,new_features

def rank_features(vec,all_features):
    rank = np.zeros(np.shape(vec)[0])
    for i in range(np.shape(vec)[0]):
        rank[i] = i + 1
    
    rank = pd.DataFrame(rank).T
    rank.columns = all_features
    
    return rank

def load_dataset_rois(dataset_name):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + dataset_name + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + dataset_name + ".csv" # Windows
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    return df

def find_used_rois(df,no_iterations):
   #numeros = np.zeros(np.shape(df)[1])
    final_count = []
    for z in range(np.shape(df)[0]):
        rois = []
        for i in range(np.shape(df)[1]):
            #numeros[i] = df.iat[1,i]
            rois.append(df.iat[z,i])
            
        #sor = np.sort(numeros)
        #lista_excluidos = []
        
    #    for i in range(np.shape(df)[1]):
    #        if i in numeros:
    #            t = 0
    #        else:
    #            lista_excluidos.append(i)
        
        # Get the most used rois
        lista_numeros = []
        for i in range(0,np.shape(df)[1]):
            if rois[i][-1].isnumeric():
                nu = rois[i][-1]
            if rois[i][-2].isnumeric():
                nu = rois[i][-2:]
            if rois[i][-3].isnumeric():
                nu = rois[i][-3:]
               
            lista_numeros.append(nu)     
                  
        lista_num = np.array(lista_numeros)
        #plt.hist(sorted(lista_num), color = 'blue',edgecolor = 'black',bins = 200)
        
        lista_num_int = np.zeros(len(lista_num))
        for i in range(len(lista_num)):
            lista_num_int[i] = int(lista_num[i])        
            
        final_count.append(lista_num_int)
    
    return final_count

def freq_appearance_rois(list_rois,number_of_rois,no_iterations_avg):    

    # Get the average frequency appearance of each roi
    avg_freq_roi = np.zeros(number_of_rois)
    for i in range(no_iterations_avg):
        list_rois_iter = list(list_rois[i])
        frequency_rois = [len(list(group)) for key, group in groupby(sorted(list_rois_iter))]        
        
        list_present_rois = []
        for j in range(number_of_rois+1):
            if j in list_rois_iter:
                list_present_rois.append(j)
         
        rois_frequency = []
        rois_frequency.append(list_present_rois)
        rois_frequency.append(frequency_rois)
                
        for j in range(len(list_present_rois)):        
            avg_freq_roi[int(list_present_rois[j]-1)] += frequency_rois[j]
            
            
    avg_freq_roi = avg_freq_roi / no_iterations_avg
    avg_freq_roi = pd.DataFrame(avg_freq_roi).T    
    avg_freq_roi.columns = np.arange(1,201)
    
    return avg_freq_roi

def get_roi_features_rank(mat,no_iterations,no_rois,names_feature_rank,avg_freq_roi):
    roi_info = []
    for i in range(no_rois):
        features_roi_freq = []
        features_roi_rank = []
        for j in range(no_iterations):    
            features_iter = mat[j,:]
            for z in range(len(features_iter)):
                ## get the roi to which the feature belong
                if features_iter[z][-1].isnumeric():
                    nu = features_iter[z][-1]
                if features_iter[z][-2].isnumeric():
                    nu = features_iter[z][-2:]
                if features_iter[z][-3].isnumeric():
                    nu = features_iter[z][-3:]
                    
                ## See if the roi is the one we want
                if int(nu) ==  (i+1):
                    #if features_iter[z] in features_roi_freq:
                    #    nada = 1
                    #else:
                        for y in range(len(names_rank)):
                            if names_rank[y] == features_iter[z]:
                                features_roi_rank.append(y+1)
                                features_roi_freq.append(features_iter[z])
                                break
        l = []
        l.append(features_roi_rank)
        l.append(features_roi_freq)
        lm = np.mean(features_roi_rank)
        l.append(lm)
        lf = avg_freq_roi.iat[0,i]
        l.append(lf)
        roi_info.append(l) 
        
    return roi_info

def plot_bubbles(rois_info,thresold,color:str="Blues",name_plot:str=""):
    # Get the data
    x = np.zeros(no_rois)
    y = np.zeros(no_rois)
    size = np.zeros(no_rois)
    
    x = np.zeros(no_rois)
    y = np.zeros(no_rois)
    size = np.zeros(no_rois)
            
    for i in range(no_rois):
        x[i] = i+1        
        if rois_info[i][3] >= threshold:
            y[i] = rois_info[i][2] 
            size[i] = rois_info[i][3]
    
    # Set figure size 
    plt. figure(figsize=(25,15))
    # Scatterplot
    plt.scatter(
        x = x,
        y = y,
        s = size*200,
        c = size,
        cmap= color, ## Oranges,Blues,Greens,Purples,Reds
        alpha = 0.6,
        edgecolors = "white", 
        linewidth = 2
        )
    
    # Add titles (main and on axis)    
    plt.xlabel("ROI",size=20)
    plt.ylabel("Average Rank",size=20)
    plt.title(name_plot,size=30)
    plt.gca().invert_yaxis()
    plt.gca().set_ylim([1800, 0])
    plt.xlim(0, 200);
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt

#%% Main programm
import matplotlib.pyplot as plt
## Load the data
csv_folder = "CSV\\"
dataset_name = "features_male_control_vs_allcausal_features"
df = pd.read_csv( csv_folder + dataset_name + ".csv")
mat = np.array(df)

### Each feature statistics (individual)
vec,all_features = frequency_appearance_avg_rank_variance(mat)
# Sort the features and its characteristics according to the most important ones (First column= repetition second= importance *lower is better* 3rd variance)
vec,all_features = sort_all_features(vec,all_features)
rank_features_var = rank_features(vec,all_features)
names_rank = rank_features_var.columns 

### ROIS repetition and rank of the features in that ROI
df = load_dataset_rois(dataset_name)
no_iterations = 10 # number of causal forest iterations
no_rois = 200 # Number of regions of interest in the brain 

list_rois = find_used_rois(df,no_iterations)
avg_freq_roi = freq_appearance_rois(list_rois,no_rois,no_iterations)

### Create a list of lists with the features that exist in each ROI and their ranking
# List has this info: 
#    [0] ranking of the feature
#    [1] feature
#    [2] Avg_ranking of the features
#    [3] Avg frequency of the roi 
rois_info = get_roi_features_rank(mat,no_iterations,no_rois,names_rank,avg_freq_roi)
threshold = 4

fig_1 = plot_bubbles(rois_info,threshold,color="Greens",name_plot="Male control")

dataset_name = "features_female_control_vs_allcausal_features"
df = pd.read_csv (csv_folder + dataset_name + ".csv")
mat = np.array(df)

### Each feature statistics (individual)
vec,all_features = frequency_appearance_avg_rank_variance(mat)
# Sort the features and its characteristics according to the most important ones (First column= repetition second= importance *lower is better* 3rd variance)
vec,all_features = sort_all_features(vec,all_features)
rank_features_var = rank_features(vec,all_features)
names_rank = rank_features_var.columns 

### ROIS repetition and rank of the features in that ROI
df = load_dataset_rois(dataset_name)
no_iterations = 10 # number of causal forest iterations
no_rois = 200 # Number of regions of interest in the brain 

list_rois = find_used_rois(df,no_iterations)
avg_freq_roi = freq_appearance_rois(list_rois,no_rois,no_iterations)

rois_info = get_roi_features_rank(mat,no_iterations,no_rois,names_rank,avg_freq_roi)
threshold = 4

fig_2 = plot_bubbles(rois_info,threshold,color="Reds",name_plot="Female control")


dataset_name = "features_male_PD_vs_allcausal_features"
df = pd.read_csv(csv_folder + dataset_name + ".csv")
mat = np.array(df)

### Each feature statistics (individual)
vec,all_features = frequency_appearance_avg_rank_variance(mat)
# Sort the features and its characteristics according to the most important ones (First column= repetition second= importance *lower is better* 3rd variance)
vec,all_features = sort_all_features(vec,all_features)
rank_features_var = rank_features(vec,all_features)
names_rank = rank_features_var.columns 

### ROIS repetition and rank of the features in that ROI
df = load_dataset_rois(dataset_name)
no_iterations = 10 # number of causal forest iterations
no_rois = 200 # Number of regions of interest in the brain 

list_rois = find_used_rois(df,no_iterations)
avg_freq_roi = freq_appearance_rois(list_rois,no_rois,no_iterations)


rois_info = get_roi_features_rank(mat,no_iterations,no_rois,names_rank,avg_freq_roi)
threshold = 4

fig_3 = plot_bubbles(rois_info,threshold,color="Purples",name_plot="Male PD")

dataset_name = "features_female_PD_vs_allcausal_features"
df = pd.read_csv(csv_folder + dataset_name + ".csv")
mat = np.array(df)

### Each feature statistics (individual)
vec,all_features = frequency_appearance_avg_rank_variance(mat)
# Sort the features and its characteristics according to the most important ones (First column= repetition second= importance *lower is better* 3rd variance)
vec,all_features = sort_all_features(vec,all_features)
rank_features_var = rank_features(vec,all_features)
names_rank = rank_features_var.columns 

### ROIS repetition and rank of the features in that ROI
df = load_dataset_rois(dataset_name)
no_iterations = 10 # number of causal forest iterations
no_rois = 200 # Number of regions of interest in the brain 

list_rois = find_used_rois(df,no_iterations)
avg_freq_roi = freq_appearance_rois(list_rois,no_rois,no_iterations)

rois_info = get_roi_features_rank(mat,no_iterations,no_rois,names_rank,avg_freq_roi)
threshold = 4

fig_4 = plot_bubbles(rois_info,threshold,color="Oranges",name_plot="Female PD")

plt.show()


