#Main causal feature selection
#XGBoost
#%% Classification separated by IMF 
#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
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

import functions_feature_selection as ffs
import functions_paths as fp

def load_dataset(dataset_name,linux:bool=False):
    path_name = fp.path_CSV(linux=linux)
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
    
    if "Gender" in df:
        df = df.drop('Gender', axis=1)
    
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
    min_max_scaler = MinMaxScaler()
    xtrain = min_max_scaler.fit_transform(xtrain)
    #scaler = preprocessing.StandardScaler()
    #xtrain = scaler.fit_transform(xtrain)
    
    return xtrain,ytrain,feature_list,removed_features


def create_n_folds(x,y,n_splits:int=5,shuffle:bool=True):
    list_xtrain_folds = []
    list_xtest_folds = []
    list_ytrain_folds = []
    list_ytest_folds = []
    
    kf = KFold(n_splits=n_splits,random_state=12, shuffle=shuffle)
    kf.get_n_splits(x)    
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        list_xtrain_folds.append(X_train)
        list_xtest_folds.append(X_test)
        list_ytrain_folds.append(y_train)
        list_ytest_folds.append(y_test)
        
    return list_xtrain_folds,list_xtest_folds,list_ytrain_folds,list_ytest_folds
    

## Get the metrics from cross validation once you have a trained algorithm 
def cross_validation(X,y,classifier,cv:int=10,n_jobs:int=-1):
    
    ## Get the scores metrics
    scoring = ['accuracy','f1_macro','precision_macro','recall_macro','roc_auc']
    scores = cross_validate(classifier,X,y,scoring=scoring,cv=cv,n_jobs=n_jobs)
    
    mean_accuracy = np.mean((scores.get("test_accuracy")[:]))
    mean_f1 = np.mean(scores.get("test_f1_macro")[:])
    mean_precision = np.mean(scores.get("test_precision_macro")[:])
    mean_recall = np.mean(scores.get("test_recall_macro")[:])
    mean_roc_auc = np.mean(scores.get("test_roc_auc"))

    return mean_accuracy,mean_f1,mean_precision,mean_recall,mean_roc_auc


    
    