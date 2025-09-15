# Overview
This project consists of two main parts:
1. **Data simulation and performance evaluation**
2. **Method application to real data**

# Code Sources
Code sources are documented in the dissertation.  
Any external sources without a license have been removed and replaced with placeholders.  
Confidential datasets and large raw files have also been removed to maintain a clear and shareable structure.

# General Structure
This is the general project structure, with the names of removed files included for clarity:

sim_eval/
|-- configs/
|   |-- lahme2.yml
|   \-- scmenv1.yml
|-- data/
|   |-- correct_coding.R   # script provided with dataset
|   |-- extract_data.R     # script provided with dataset
|   |-- Oxford Trust Survey_FINAL DATA_adj2.sav   # original dataset (removed)
|   |-- new_cleaning.Rmd   # data processing
|   |-- original_data.csv  # original ordinal and binary variables
|   |-- summed_all.csv     # age and summed-up variables 
|   \-- summed_demo.csv    # summed variables with age and income
|-- DataAnalysis/
|   |-- Bootstrap/         # results from bootstrapping runs 
|   |-- bootstrap.py       # run bootstrapping
|   |-- bootstrapanalysis.py   # combine bootstrapping results
|   |-- MainResults/       # main results storage 
|   |-- running_real_data.py   # RLCD on ordinal data
|   |-- subsampleanalysis.py   # combine subsampling results 
|   |-- Subsampling/       # results from subsampling runs 
|   |-- subsampling.py     # run subsampling 
|   |-- SummedAnalysis.py  # RLCD on summed data
|   \-- ...                # helpers, CPDAG conversion, graphing
|-- simulation_summary/ 
|   |-- combine_continuous.r   # combine remote and local results
|   |-- combine_thresholds.py  # combine individual results 
|   |-- thresholds/            # thresholding results 
|   \-- ...                    # result files 
|-- lahme/     # LaHME method implementation
|-- scm/       # RLCD method implementation
|-- true_DAGs/ # true DAGs for evaluation
|-- convert_to_CPDAG.py        # CPDAG conversion helpers 
|-- DataModel.py               # simulation model (same as in scm)
|-- GraphDrawer.py             # simple graphing (same as in scm)
|-- LinearSCM.py               # linear SCM simulation (same as in scm)
|-- metrics_revised.py         # metric calculation functions
|-- r_only.py                  # wrapper to run PC and GES in R
|-- sim_eval_combined_remote_laplace.py   # simulation with Laplace noise 
|-- sim_eval_combined_remote.py           # simulation with Gaussian noise 
|-- sim_threshold_gaussian.py             # simulation with thresholding
|-- simulation_scenarios.py   # seven evaluation scenarios
|-- test_scenarios.py         # convert true DAG to graph 
\-- ...                       # worker scripts and misc. files