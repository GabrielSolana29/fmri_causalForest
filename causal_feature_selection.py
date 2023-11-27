## Causal Feature Selection and wrapper feature subset selection
# import libraries
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
linux = False
import os

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
    
import functions_causal_feature_selection as fcfs
import functions_feature_selection as ffs
import functions_paths as fpt
import functions_assist as fa
#%% Functions

def load_dataset(dataset_name,linux:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + dataset_name + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + dataset_name + ".csv" # Windows
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    
    df = df
    class_vec = df.Class
    subject_vec = df.Id
        
    df = df.drop('Id', axis=1)
    df = df.drop('Class', axis=1)
    
    if len(dataset_name) > 8:
        df = df.drop('Gender',axis=1)
    
    return class_vec,subject_vec,df

def load_preprocess_dataset(name_csv,variance_t:bool=False,remove_feature_list:list=[],linux:bool=False):
    ytrain,subject_vec,xtrain = load_dataset(name_csv,linux = linux)
    feature_list = xtrain.columns 
    xtrain = np.array(xtrain)       
    removed_features = []    
    ### Pre processing 
    if variance_t ==True:
        # remove features with 0 variance 
        xtrain,features_removed = ffs.variance_threshold(xtrain)
        # Remove the items from the lists        
        feature_list = list(feature_list)
        for i in range(len(features_removed)):
            removed_features.append(feature_list[features_removed[i]])
            
    if len(remove_feature_list):           
        df = pd.DataFrame(xtrain)        
        df.columns = feature_list[:-1]        
        for i in range(len(feature_list)):            
            if feature_list[i] in remove_feature_list:
                df = df.drop(feature_list[i],1)
                                  
        xtrain = np.array(df)
        feature_list = list(df.columns)
        feature_list.append("target")                             
            
    #transformer = RobustScaler().fit(xtrain)
    #xtrain = transformer.transform(xtrain)
    min_max_scaler = preprocessing.MinMaxScaler()
    xtrain = min_max_scaler.fit_transform(xtrain)
    #scaler = preprocessing.StandardScaler()
    #xtrain = scaler.fit_transform(xtrain)
    
    return xtrain,ytrain,feature_list,removed_features

def test_main_wfss(name_csv,classifier,cv,n_jobs:int=-1):
    
    xtrain,ytrain,feature_list,removed_features = load_preprocess_dataset(name_csv,linux = linux)    
    xtrain = np.array(xtrain)
    
    df = pd.DataFrame(xtrain)
    df.insert(loc=np.shape(df)[1], column='target', value=ytrain)
    feature_list = feature_list.insert(np.shape(xtrain)[1],"target") 
    df.columns = feature_list
    
    feature_list_str = []
    for i in range(len(feature_list)):
        feature_list_str.append(str(feature_list[i])) 
        
    ## Shuffle the dataset 
    xtrain,ytrain = shuffle(xtrain,ytrain)
    results = []
    for i in range(1):
        list_iteration = []
        tic = time.perf_counter()          
        if i == 0:
            drop_perc = 99.5
        elif i == 1:
            drop_perc= 99.5
        elif i == 2:
            drop_perc= 99
        elif i == 3:
            drop_perc = 98
        elif i == 4:
            drop_perc = 96
            
        best_subset,feature_importance = fcfs.causal_feature_selection_pipeline(xtrain,ytrain,feature_list=feature_list_str,filter_treatment='info_gain',w_algorithm=classifier,drop_percentage=drop_perc,shap_value=False,n_jobs=n_jobs,verbose=True)
        toc = time.perf_counter()
        print("\nTime elapsed in wrapper and causal forest: ",(toc-tic), "seconds")
        
        best_subset.insert(loc=np.shape(best_subset)[1], column='target', value=ytrain)
        best_subset = best_subset.drop('target',axis=1)
        X = best_subset
        y = ytrain
        
        best_subset.insert(loc=np.shape(best_subset)[1], column='target', value=ytrain)
        best_subset = best_subset.drop('target',axis=1)
        X = best_subset
        y = ytrain
        if classifier == 'xgb':
            classifier_algo =  XGBClassifier(use_label_encoder=False)
        elif classifier == 'lr':
            classifier_algo = LogisticRegression(class_weight='balanced',solver='lbfgs',random_state=42, n_jobs=n_jobs,max_iter=1000)
        elif classifier == 'knn':
            classifier_algo = KNeighborsClassifier(n_neighbors=3)

        metrics = fa.cross_validation(X,y,classifier_algo,cv=cv,n_jobs=-1)

        list_iteration.append(classifier)
        list_iteration.append(drop_perc)
        list_iteration.append(np.shape(best_subset))
        list_iteration.append(metrics)
        res_iter = pd.DataFrame(list_iteration)
        name = name_csv + "Iter_results_" + classifier + "_" + str(i) + ".csv"
        res_iter.to_csv(name)
        results.append(list_iteration)
        print ("\n\n Finished drop ", drop_perc, " perc iteration\n\n")
        
    return results

#%% Load data
tic = time.perf_counter()
name_csv = "features"
xtrain,ytrain,feature_list,removed_features = load_preprocess_dataset(name_csv,linux = linux)
xtrain = np.array(xtrain)

### differente feature selection methods to find the most impactful feature
fisher_r = ffs.fisher_rank(xtrain,ytrain,plot=True)
mutual_info_gain_r = ffs.mutual_information_gain_rank(xtrain,ytrain,plot=True)
mad_r = ffs.mean_absolute_difference(xtrain, plot=True)

feature_list[fisher_r[0]]
feature_list[mutual_info_gain_r[0]]
feature_list[mad_r[0]]

toc = time.perf_counter()
print("\nTime elapsed: ",(toc-tic), "seconds")

#%% Create data frame with the features and target for causal feature selection 
df = pd.DataFrame(xtrain)
df.insert(loc=np.shape(df)[1], column='target', value=ytrain)
feature_list = feature_list.insert(np.shape(xtrain)[1],"target") 
df.columns = feature_list

#%% Causal forest feature selection
tic = time.perf_counter()
feature_importance = fcfs.causal_forest_feature_selection(df, "target",n_estimators = 10000, most_important_feature = feature_list[mutual_info_gain_r[0]], shap_values=False, optimize_parameters=True)
toc = time.perf_counter()
print("\nTime elapsed in causal forest: ",(toc-tic), "seconds")

#%% Complete pipeline with wrappers for feature subset selection 
feature_list_str = []
for i in range(len(feature_list)):
    feature_list_str.append(str(feature_list[i])) 
    
## Shuffle the dataset 
xtrain,ytrain = shuffle(xtrain,ytrain)
    
tic = time.perf_counter()  
classifier = 'xgb'
# Select the percentage of features from the ranking that should be returned
drop_perc= 99
n_jobs = -1
best_subset,feature_importance = fcfs.causal_feature_selection_pipeline(xtrain,ytrain,feature_list=feature_list_str,filter_treatment='info_gain',w_algorithm=classifier,drop_percentage=drop_perc,shap_value=False,n_jobs=n_jobs,verbose=True)
toc = time.perf_counter()
print("\nTime elapsed in wrapper and causal forest: ",(toc-tic), "seconds")

