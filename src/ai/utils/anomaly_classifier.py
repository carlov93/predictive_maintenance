import numpy as np
import pandas as pd


class AnomalyClassifier:
    """
    This class implements methods for classifying sensor measurements depending on the selected
    novelty metric and to retrieve the reconstruction error from the training data set.
    """
    @staticmethod
    def apply_prediction_interval_metric(results_prediction, no_sensors, standard_deviation):
        """
        This method classifies sensor measurements as an anomaly (1) if the sensor value is
        higher or lower than the given uncertainty interval, otherwise 0. This uncertainty interval
        is data dependent (differs between each sample and each sensor).
        It is important, that the data set has the following structure regarding the columns:
        1. ID
        2. Targets for each sensor
        3. Mu for each sensor
        4. Sigma for each sensor
        :param results_prediction:
        :param no_sensors:
        :param standard_deviation:
        :return:
        """
        # Drop all empty columns
        unnamed_columns = results_prediction.columns.str.contains('unnamed', case=False)
        results_prediction.drop(results_prediction.columns[unnamed_columns], axis=1, inplace=True)
        
        for i in range(1, no_sensors+1):
            sensor_target = results_prediction.iloc[:, i].values
            sensor_mu = results_prediction.iloc[:, i+no_sensors].values
            sensor_sigma = results_prediction.iloc[:, i+2*no_sensors].values

            anomaly = []
            for target_i, mu_i, sigma_i in zip(sensor_target, sensor_mu, sensor_sigma):
                if target_i < (mu_i - standard_deviation * sigma_i):
                    anomaly.append(1)
                elif target_i > (mu_i + standard_deviation * sigma_i):
                    anomaly.append(1)
                else:
                    anomaly.append(0)
            
            results_prediction["Anomaly Sensor_"+str(i)] = anomaly
        return results_prediction

    @staticmethod
    def apply_euclidean_distance_metric(results_prediction, no_sensors, threshold_machine, percentage, each_sensor=False, threshold_sensors=[]):
        """
        This method classifies sensor measurements as an anomaly (1) if loss is higher
        than given threshold, otherwise 0. This method is for classifying predictions
        to normal or anomalous points given a threshold for the hole machine or for each
        sensor individually.
        It is important, that the data set has the correct structure regarding the columns:
        1. ID
        2. Targets for each sensor
        3. Prediction for each sensor
        4. Reconstruction error for each sensor
        :param results_prediction:
        :param no_sensors:
        :param threshold_machine:
        :param percentage:
        :param each_sensor:
        :param threshold_sensors:
        :return:
        """
        # Drop all empty columns
        unnamed_columns = results_prediction.columns.str.contains('unnamed', case=False)
        results_prediction.drop(results_prediction.columns[unnamed_columns], axis=1, inplace=True)

        if each_sensor:
            for i in range(1, no_sensors+1):
                results_prediction["Anomaly Sensor_"+str(i)] = np.where(results_prediction.iloc[:, i+2*no_sensors] >= threshold_sensors[i-1]*percentage, 1, 0)
            return results_prediction
        else:
            for i in range(1, no_sensors+1):
                results_prediction["Anomaly Sensor_"+str(i)] = np.where(results_prediction.iloc[:, i+2*no_sensors] >= threshold_machine*percentage, 1, 0)
            return results_prediction

    @staticmethod
    def get_threshold(path_training_data, no_sensors, each_sensor=True):
        """
        This method retrieves the mean reconstruction error of a given training data set.
        It is calculated for each sensor separatly or for the entire machine depending on the parameter.
        :param path_training_data:
        :param no_sensors:
        :param each_sensor:
        :return:
        """
        results_prediction = pd.read_csv(path_training_data, sep=";")
        no_samples = results_prediction.shape[0]
        reconstruction_errors = results_prediction.iloc[:, 1+2*no_sensors:1+3*no_sensors]
        
        mean_RE_per_sensor = reconstruction_errors.sum(axis=0) / no_samples
        threshold_mean = mean_RE_per_sensor.tolist()
        max_RE_per_sensor = reconstruction_errors.max(axis=0)
        threshold_max = max_RE_per_sensor.tolist()
            
        if not each_sensor:
            threshold_mean = threshold_mean.sum() / no_sensors
            threshold_max = threshold_max.sum() / no_sensors
            
        return threshold_mean, threshold_max
