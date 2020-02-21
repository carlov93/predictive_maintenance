# 0. Introduction
The early detection and interpretation of novelties are important because it provides information about an acute or imminent critical event. In the context of industrial plants, these events are faulty behaviors of machine parts, which can lead to damage or decreasing product quality. Due to the ever-increasing complexity of industrial plants, it is hardly possible to detect inconsistencies in machine behavior manually. The data in this domain has a temporal aspect, typically due to the dynamics of a machine. As a consequence, types of anomalies are mostly of contextual or collective nature. The chosen ML has to be able to model temporal patterns.

## General Approach
This master thesis introduces a three step approach, illustrated by figure 17. In the first step, a model of normality is trained in an unsupervised manner on a data set containing sensor signals representing the CPS’s normal behavior. I choose a LSTM able to model internal dynamics with the cell state in order to learn the behavior of dynamic systems. These kinds of NN should be able to detect contextual anomalies in time series data of CPS. The coverage of temporal patterns over a long period is necessary because the information from several time steps ago is relevant for predicting the next sensor value of a dynamic machine. If the model can do so, the model’s predicted value is the expected value for the normal behavior of a CPS.

In the second step, the model receives a sequence of past observations and predicts the next sensor signal. This procedure is similar to the reconstruction-based approach. The prediction of sensor values is just an interim step; the goal is the classification of a measure into normal or abnormal. 

In the third step, one compares the actual sensor values $x_{t+1}$ of the CPS with the predicted value $\hat{x}_{t+1}$ in order to detect novel behavior. If degradation occurs, a deviation between prediction and observation should be noticeable because the real sensor values drift away from the predictions. Thus, one should be able to detect contextual anomalies in the subsequent classification. A threshold value must be defined up to which a deviation is still considered normal.
 


# 1. Project Setup
## AW SageMaker
1. Navigate to project root directory:
    - ``` cd SageMaker/predictive_maintenance```
2. Activate virtual environment:
    - `source venv/bin/activate`
3. Add virtual environment as a kernel 
    - `ipython kernel install --user --name=masterarbeit`

## Local 
1. Navigate to project root directory
2. Create virtual environment: 
    - `python3 -m venv venv_cm`
3. Activate virtual environment:
    - `source venv_cm/bin/activate`
4. Install ipykernel to add virtual environment to jupyter lab: 
    - `pip install ipykernel`
    - `ipython kernel install --user --name=condition_monitoring`
5. Install libraries from requirements.txt
6. Add path of own packages to virtual environment
    - Create a file with the name _sitecustomize.py_ in this directory: `venv_cm/lib/python3.6/site-packages/`
    - Insert this code to the file: 
    
```pyhton
import os
import sys

sys.path.append("<Pfad>/src/ai/models/")
sys.path.append("<Pfad>/src/ai/utils/")
```

# 2. Overview of Directory
- _data_: raw data of artifical and real world data sets
- _ideas_: some scripts that contain first ideas of further work
- _models_: trained LSTM models are stored here as well as log-files of training process
- _src_
    - _ai_: all scripts regarding model training, cross-validation, prediction, classification, helper classes, ...
    - _visualisation_: csv files of classification results, visualisation scripts and evaluation scripts are stored here


# 3. Steps to execute novelty detection 
1. Train model of normality:
    1. Cross-validation and grid search
        - Execute Cross Validation in order to check ability of generalisation and to find best hyperparameter: src/ai/train_model/cross_validation/
        - Results are stored in: models/cross_validation/
    2. Train final model with best hyperparameters: 
        - Execute train_final_MLE_model.ipynb in src/ai/train_model/
        - Trained model and log-file is stored in: models/MLE/
2. Predict sensor values with trained model
    1. Copy mean and variance from training process to predict_sensor.ipynb in order to scale the data acordingly 
    2. Run the predict_sensor.ipynb notebook (initialise MSE or MLE model)
    3. Results are stored here: src/visualisation/files/prediction/
3. Classifiy sensor measurements by comparing predicted and true sensor values
    1. Run the classify_sensor.ipynb notebook
    2. Results are stored here: src/visualisation/files/classification/
4. Evaluation and Visualisation of results
    1. The visualization methods may have to be adapted to the data set