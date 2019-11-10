This folder includes files for application to AI Singapore Apprenticeship Programme for Khalishah Nadhirah Bte Abu Bakar (NRIC ending 1910G)

Files include
(1) mlp folder
  - Contains 2 files
    (i) mlp_proj_working_file.ipynb - File where I worked on the model training and model tuning
    (ii) mlp_modules.ipynb - File which contains functions to run the machine learning pipeline
      - Function 1 - data_prep()
      - Function 2 - traffic_model(df)
      (elaboration of functions below)
(2) README.md
  - Description of files 
(3) eda.ipynb
  - This file contains the exploratory data analysis of traffic_data.csv.
(4) requirements.txt 
(5) run.sh

mlp_modules.ipynb contains:-
(1) data_prep() function executes the following:
- Import necessary packages
- Read data from URL
- Parse dates in dataframe and extract date features from date field
- Fixing in accuracies in data like strings case, granularity of dataframe
- Extract information from Holiday feature
- Encoding of selected categorical features
- Drop unnecessary features
- Returns final dataframe to be fed into the model

(2) traffic_model(df) executes the following, taking cleaned dataframe as an input:
- Import necessary packages
- Get independent and dependent variables
- Train Test Validation Split
- Specify metrics for validation
- Specify functions to print scores
- Specify final model with final chosen training parameters
