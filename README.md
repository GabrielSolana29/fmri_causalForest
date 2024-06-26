<h1 align="left"> Causal Forest Machine Learning Analysis of Parkinson's Disease in resting-state Functional Magnetic Resonance Imaging </h1>
<!-- Badges generated with: https://michaelcurrin.github.io/badge-generator/#/generic -->
<a href="https://www.linux.org/" title="Go to Linux homepage"><img src="https://img.shields.io/badge/OS-Linux-blue?logo=linux&logoColor=white" alt="OS - Linux"></a>
<a href="https://www.microsoft.com/" title="Go to Microsoft homepage"><img src="https://img.shields.io/badge/OS-Windows-blue?logo=windows&logoColor=white" alt="OS - Windows"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/static/v1?label=Python&message=3.10.9&color=2ea44f" alt="Python - 3.10.9"></a>

<h2 align="left"> About the project </h2>

This work presents a methodology to analyze functional Magnetic Resonance Imaging (fMRI) from Parkinson's disease (PD) patients by pre-processing the images and extracting features from the time series generated from the activations in the Regions of Interest (ROI) in the brain. Then, a proposed pipeline to perform dimensionality reduction and subset selection is used by leveraging techniques like Causal Forest and Wrapper Feature Subset Selection (WFSS). Finally, we present the relations between the ROIs and the classes with bubble plots and with Multiple Correspondence Analysis to facilitate the visualization of the relationships.

<h2 align="left"> Requirements </h2>

<h3 align="left"> Libraries </h3>

- Python 3.9
- pandas 2.0.2
- numpy 1.24.4
- scikit-learn 1.2.2
- scipy 1.11.0
- econml 0.14.1
- statsmodels 0.14.0
- xgboost 1.7.6
- prince 0.7.1
- matplotlib 3.7.1
- shap 0.41.0
- mlxtend 0.22.0

<h3 align="left"> Dataset </h3>

The data used in this project comes from fMRI obtained from the Parkinson's Progression Marker Initiative (PPMI) <a href="url">https://www.ppmi-info.org/</a>, and 1000 functional Connectomes Project <a href="url">https://www.nitrc.org/projects/fcon_1000/</a>. Even though both datasets are publicly available, the data included here does not contain any information to identify the patients, and has already gone through a pre-processing stage explained in the paper.
In the CSV folder, the following files are available:

- <b>activations.csv:</b> Contains the time series from each Region of Interest obtained with the changes in the the gray level of the fMRI from the brain of patients (class 1) and controls (class 0).
- <b>features.csv:</b> Contains the features extracted from the time series for each patient and control. In total there are 11600 features per patient.
- <b>features_female_control_vs_allcausal_features.csv:</b> Best ranked features from Causal Forest in multiple iterations for healthy female patients.
- <b>features_female_PD_vs_allcausal_features.csv:</b> Best ranked features from Causal Forest in multiple iterations for PD female patients.
- <b>features_male_control_vs_allcausal_features.csv:</b> Best ranked features from Causal Forest in multiple iterations for healthy male patients.
- <b>features_male_PD_vs_allcausal_features.csv:</b> Best ranked features from Causal Forest in multiple iterations for PD male patients.
- <b>mca_pd_control_female_male:</b> Contingency table where each row and column corresponds to a group and a brain region, respectively. 1 signals a causal effect, 0 indicates the contrary.
- <b>mca_updrs_pd_female_male:</b> Contingency table where each row and column corresponds to a group and a brain region, respectively. 1 signals a causal effect, 0 indicates the contrary.

For more information on the datasets please refer to the article.

<h2 align="left"> Feature selection with Causal Forest and Wrapper Features Subset Selection </h2>

- <b>feature_extraction.py:</b> Loads the file "activations.csv" and extracts 11600 features per patient, including statistical, frequency-based and connectivity-based features. It generates the file "features.csv".
- <b>causal_feature_selection.py:</b> Loads the file "features.csv" and performs feature selection using Causal Forest and a custom function which combines Causal Forest and WFSS, providing both the rankings of the features, and the best subset obtained by the WFSS algorithm.
- <b>bubble_plots.py:</b> It loads the files containing the best ranking features provided by multiple iterations of the Causal Forest algorithm by creating 2 classes: (1) female PD patients and the rest of the observations (2) female controls and the rest of the observations (3) male PD patients and the rest of the observations (4) male controls and the rest of the observations. To visualize the importance and frequency of appearance of the features being selected, bubble plots are generated.

<h2 align="left"> Multiple Correspondence Analisis </h2> 

- <b>multiple_correspondence_analysis.py:</b> Performs Multiple Correspondence Analysis to better visualize the relation between the ROI and the classes. Loads a continency table from one of the two provided files ("mca_pd_control_female_male.csv" or "mca_updrs_pd_female_male.csv").


<!-- ## License 

This project is [GNU licensed](./LICENSE). -->
