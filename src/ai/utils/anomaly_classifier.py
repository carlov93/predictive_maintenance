import numpy as np
import pandas as pd

class AnomalyClassifier():        
    def apply_prediction_interval_metric(self, results_prediction, no_sensors, standard_deviation):
        """
        This method is for predictions from LSTM Model with mean and variance prediction. 
        It is important, that the dataframe has the correct structure:
        1. ID
        2. Targets for each sensor
        3. Mu for each sensor
        4. Sigma for each sensor
        """
        # Drop all empty columns
        unnamed_columns = results_prediction.columns.str.contains('unnamed',case = False)
        results_prediction.drop(results_prediction.columns[unnamed_columns],axis = 1, inplace = True)
        
        for i in range(1, no_sensors+1):
            sensor_target = results_prediction.iloc[:, i].values
            sensor_mu = results_prediction.iloc[:, i+no_sensors].values
            sensor_sigma = results_prediction.iloc[:, i+2*no_sensors].values
            
            # tag sample as an anomaly (1) if sensor value is higher or lower than given confidence band, otherwhise 0 
            anomaly = []
            for target_i, mu_i, sigma_i in zip(sensor_target, sensor_mu, sensor_sigma):
                if target_i < (mu_i - standard_deviation * sigma_i):
                    anomaly.append(1)
                elif target_i > (mu_i + standard_deviation * sigma_i):
                    anomaly.append(1)
                else:
                    anomaly.append(0)
            
            results_prediction["Anomaly Sensor_"+str(i)]= anomaly
        return results_prediction
        
    def apply_euclidean_distance_metric(self, results_prediction, no_sensors, threshold_machine, percentage, each_sensor=False, threshold_sensors=[]):
        """
        This method is for predictions from LSTM Model with mean prediction. 
        This method is for classifiying predictions to normal or anomalous points given a threshold for the hole machine.
        It is important, that the dataframe has the correct structure:
        1. ID
        2. Targets for each sensor
        3. Prediction for each sensor
        4. Reconstruction error for each sensor
        """
        # Drop all empty columns
        unnamed_columns = results_prediction.columns.str.contains('unnamed',case = False)
        results_prediction.drop(results_prediction.columns[unnamed_columns],axis = 1, inplace = True)
        
        # tag sample as an anomaly (1) if loss is higher than given threshold, otherwhise 0 
        if each_sensor:
            for i in range(1, no_sensors+1):
                results_prediction["Anomaly Sensor_"+str(i)]=np.where(results_prediction.iloc[:,i+2*no_sensors]>=threshold_sensors[i-1]*percentage, 1, 0)

            return results_prediction
        else:
            for i in range(1, no_sensors+1):
                results_prediction["Anomaly Sensor_"+str(i)]=np.where(results_prediction.iloc[:,i+2*no_sensors]>=threshold_machine*percentage, 1, 0)

            return results_prediction
        
    def get_threshold(self, path_training_data, no_sensors, each_sensor=True):
        results_prediction = pd.read_csv(path_training_data, sep=";")
        no_samples = results_prediction.shape[0]
        reconstruction_errors = results_prediction.iloc[:,1+2*no_sensors:1+3*no_sensors]
        
        mean_RE_per_sensor = reconstruction_errors.sum(axis=0) / no_samples
        threshold_mean = mean_RE_per_sensor.tolist()
        
        max_RE_per_sensor = reconstruction_errors.max(axis=0)
        threshold_max = max_RE_per_sensor.tolist()
            
        if not each_sensor:
            threshold_mean = threshold_mean.sum() / no_sensors
            threshold_max = threshold_max.sum() / no_sensors
            
        return threshold_mean, threshold_max
               