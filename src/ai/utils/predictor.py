import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math


class PredictorMse:
    """
    This class implements methods to predict new sensor data with a trained ML model
    uses the MSE loss function.
    """
    def __init__(self, model, path_data, columns_to_ignore):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore

    def get_column_names_data(self):
        """
        This method gets the column names (sensor names) of the given data set.
        :return: List of strings with column names.
        """
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n', '')
            return header.split(",")
        
    def create_column_names_result(self):
        """
        This methods creates column names for the result data set that contains the predictions.
        :return: List of strings with column names.
        """
        column_names_target = [column_name+" target" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_loss_per_sensor = [column_name+" reconstruction error " for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        
        column_names = ["ID"] + column_names_target + column_names_predicted + column_names_loss_per_sensor
        return column_names
          
    def predict(self, input_data, target_data):
        """
        This method predicts the next sensor value for each sensor given a multivariate time series
        of previous sensor values. Furthermore, it calculates the reconstruction error (RE) for each sensor.
        The RE is the euclidean distance between the target value and the predicted value.
        It is important that the sample ID is on the dimension 0 when using this method.
        :param input_data: DataLoader class provided by Pytorch.
        :param target_data: DataLoader class provided by Pytorch.
        :return: Results for given mini-batch: [[sample_1], [sample_2], ...]
        """
        self.model.eval()
        with torch.no_grad():
            id_target = target_data[:, 0]
                
            # De-select ID feature in both input_data and target data for inference
            input_data = torch.from_numpy(input_data.numpy()[:, :, 1:])
            target_data = torch.from_numpy(target_data.numpy()[:, 1:])

            # Prediction
            hidden = self.model.init_hidden()
            output = self.model(input_data, hidden)
            
            batch_results = []
            for batch in range(self.model.batch_size):
                # Reshape and calculate prediction metrics
                predicted_data = output[batch, :].data.numpy().tolist()
                ground_truth = target_data[batch, :].data.numpy().tolist()
                reconstruction_error_per_sensor = [math.sqrt((ground_truth_i - predicted_i)**2) for ground_truth_i, predicted_i in zip(ground_truth, predicted_data)]

                data = [id_target[batch].item()] + ground_truth + predicted_data + reconstruction_error_per_sensor
                batch_results.append(data)
        return batch_results

    
class PredictorMle:
    """
    This class implements methods to predict new sensor data with a trained ML model
    uses the log-likelihood loss function.
    """
    def __init__(self, model, path_data, columns_to_ignore):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore

    def get_column_names_data(self):
        """
        This method gets the column names (sensor names) of the given data set.
        :return: List of strings with column names.
        """
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n', '')
            return header.split(",")
        
    def create_column_names_result(self):
        """
        This methods creates column names for the result data set that contains the predictions.
        :return: List of strings with column names.
        """
        column_names_target = [column_name+" target" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_mu_predicted = [column_name+" mu predicted" for column_name in self.column_names_data  if column_name not in self.columns_to_ignore+["ID"]]
        column_names_sigma_predicted = [column_name+" sigma predicted" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_normalised_residuals= [column_name+" normalised residual" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        return ["ID"] + column_names_target + column_names_mu_predicted + column_names_sigma_predicted + ["mean normalised residual"] + column_names_normalised_residuals
           
    def predict(self, input_data, target_data):
        """
        This method predicts the next sensor value for each sensor given a multivariate time series
        of previous sensor values. Furthermore, it calculates the reconstruction error (RE) for each sensor.
        The RE is the euclidean distance between the target value and the predicted value.
        It is important that the sample ID is on the dimension 0 when using this method.
        Because of the transformation of sigma inside the LossModuleMle (σ_t = exp(τ_t))
        we have to revert this transformation with sigma_i = exp(tau_i).
        :param input_data: DataLoader class provided by Pytorch.
        :param target_data: DataLoader class provided by Pytorch.
        :return: Results for given mini-batch: [[sample_1], [sample_2], ...]
        """
        self.model.eval()
        with torch.no_grad():
            id_target = target_data[:, 0]
                
            # De-select ID feature in both input_data and target data for inference
            input_data = torch.from_numpy(input_data.numpy()[:, :, 1:])
            target_data = torch.from_numpy(target_data.numpy()[:, 1:])

            # Prediction
            hidden = self.model.init_hidden()
            mu, tau = self.model(input_data, hidden)
            sigma_batches = torch.exp(tau)
                
            # Reshape and calculate prediction metrics
            batch_results = []
            for batch in range(self.model.batch_size):
                mu_predicted = mu[batch, :].data.numpy().tolist()
                ground_truth = target_data[batch, :].data.numpy().tolist()
                sigma_predicted = sigma_batches[batch, :].data.numpy().tolist()
                normalised_residual_per_sensor = [(target_i - prediction_i) / sigma_i for target_i, prediction_i, sigma_i in zip(ground_truth, mu_predicted, sigma_predicted)]
                normalised_residual = sum(normalised_residual_per_sensor) / self.model.input_dim
                    
                data = [id_target[batch].item()] + ground_truth + mu_predicted + \
                       sigma_predicted + [normalised_residual] + normalised_residual_per_sensor
                batch_results.append(data)
        return batch_results
