#%% import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# pip install prince==0.7.1
import prince
linux = False
import os
#%% Import scripts
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
#%% functions
def load_dataset(dataset_name,linux:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + dataset_name + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + dataset_name + ".csv" # Windows
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])

    return df

#%% Main
#name_csv = "mca_pd_control_female_male"
name_csv = "mca_updrs_pd_female_male"
df = load_dataset(name_csv)

names = df.columns
df_arr = np.array(df)

mca = prince.MCA(n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    random_state=42
    )
mca = mca.fit(df)

ax = mca.plot_coordinates(
     X=df,
     ax=None,
     figsize=(15, 15),
     show_row_points=True,
     row_points_size=350,
     show_row_labels=True,
     show_column_points=True,
     column_points_size=100,
     show_column_labels=False,
     legend_n_cols=6
     )

ax.plot()

col = mca.col_masses_
coord = mca.column_coordinates(df)
eigen = mca.eigenvalues_
explained_inertia = mca.explained_inertia_
mass = mca.col_masses_
sort_col = np.argsort(col)

