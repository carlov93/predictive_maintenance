import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins
import math

class PredictorMse():
    def __init__(self, model, path_data, columns_to_ignore):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")
        
    def create_column_names_result(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data 
                               if column_name not in self.columns_to_ignore+["ID"]]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data 
                                  if column_name not in self.columns_to_ignore+["ID"]]
        column_names_loss_per_sensor = [column_name+" reconstruction error " for column_name in self.column_names_data 
                                        if column_name not in self.columns_to_ignore+["ID"]]
        
        column_names = ["ID"] + column_names_target + column_names_predicted + column_names_loss_per_sensor
        return column_names
          
    def predict(self, input_data, target_data):
        self.model.eval()
        with torch.no_grad():
            # Store ID of target sample 
            id_target = target_data[:,0]   #ID must be on first position!
                
            # De-select ID feature in both input_data and target data for inference
            input_data = torch.from_numpy(input_data.numpy()[:,:,1:])   # ID must be on first position!
            target_data = torch.from_numpy(target_data.numpy()[:,1:])   # ID must be on first position!

            # Initilize Hidden and Cell State
            hidden = self.model.init_hidden()

            # Forward propagation
            output = self.model(input_data, hidden)
            
            batch_results= []
            for batch in range(self.model.batch_size):
                # Reshape and Calculate prediction metrics
                predicted_data = output[batch,:].data.numpy().tolist()
                ground_truth = target_data[batch,:].data.numpy().tolist()
                reconstruction_error_per_sensor = [math.sqrt((ground_truth_i - predicted_i)**2) for ground_truth_i, predicted_i in zip(ground_truth, predicted_data )]

                # Add values to dataframe
                data = [id_target[batch].item()] + ground_truth + predicted_data + reconstruction_error_per_sensor
                batch_results.append(data)
                        
        return batch_results

    
class PredictorMle():
    def __init__(self, model, path_data, columns_to_ignore):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")
        
    def create_column_names_result(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data 
                               if column_name not in self.columns_to_ignore+["ID"]]
        column_names_mu_predicted = [column_name+" mu predicted" for column_name in self.column_names_data 
                                     if column_name not in self.columns_to_ignore+["ID"]]
        column_names_sigma_predicted = [column_name+" sigma predicted" for column_name in self.column_names_data 
                                        if column_name not in self.columns_to_ignore+["ID"]]
        column_names_normalised_residuals= [column_name+" normalised residual" for column_name in self.column_names_data 
                                            if column_name not in self.columns_to_ignore+["ID"]]
        return ["ID"] + column_names_target + column_names_mu_predicted + column_names_sigma_predicted + ["mean normalised residual"] +  column_names_normalised_residuals
           
    def predict(self, input_data, target_data):
        self.model.eval()
        with torch.no_grad():
            # Store ID of target sample 
            id_target = target_data[:,0]   #ID must be on first position!
                
            # De-select ID feature in both input_data and target data for inference
            input_data = torch.from_numpy(input_data.numpy()[:,:,1:])   # ID must be on first position!
            target_data = torch.from_numpy(target_data.numpy()[:,1:])   # ID must be on first position!

            # Initilize Hidden and Cell State
            hidden = self.model.init_hidden()

            # Forward propagation
            mu, tau = self.model(input_data, hidden)
                
            # Because of the transformation of sigma inside the LossModuleMle (σ_t = exp(τ_t))
            # we have to revert this transformation with exp(tau_i).
            sigma_batches = torch.exp(tau)
                
            # Reshape and Calculate prediction metrics
            batch_results= []
            for batch in range(self.model.batch_size):
                mu_predicted = mu[batch,:].data.numpy().tolist()
                ground_truth = target_data[batch,:].data.numpy().tolist()
                sigma_predicted = sigma_batches[batch,:].data.numpy().tolist()
                normalised_residual_per_sensor = [(target_i - prediction_i) / sigma_i for target_i, prediction_i, sigma_i in zip(ground_truth, mu_predicted, sigma_predicted)]
                normalised_residual = sum(normalised_residual_per_sensor) / self.model.input_dim
                    
                data = [id_target[batch].item()] + ground_truth + mu_predicted + sigma_predicted + [normalised_residual] + \
                normalised_residual_per_sensor
                batch_results.append(data)

        return batch_results
