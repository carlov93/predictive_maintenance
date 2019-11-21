import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins

class PredictorMleCellState():
    def __init__(self, model, path_data, columns_to_ignore, no_standard_deviation):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore
        self.no_standard_deviation = no_standard_deviation

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
        column_names_normalised_residuals = [column_name+" normalised residual" for column_name in self.column_names_data 
                                            if column_name not in self.columns_to_ignore+["ID"]]
        column_names_cell_state = ["cell_state_"+str(i) for i in range(self.model.n_hidden_lstm)]
        
        return ["ID"] + column_names_target + column_names_mu_predicted + column_names_sigma_predicted + ["mean normalised residual"] + column_names_normalised_residuals + column_names_cell_state
           
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
            cell_state_batch = self.model.current_cell_state
                
            # Because of the transformation of sigma inside the LossModuleMle (σ_t = exp(τ_t))
            # we have to revert this transformation with exp(tau_i).
            sigma_batches = torch.exp(tau)
                
            # Reshape and Calculate prediction metrics
            batch_results= []
            for batch in range(self.model.batch_size):
                mu_predicted = mu[batch,:].data.numpy().tolist()
                ground_truth = target_data[batch,:].data.numpy().tolist()
                sigma_predicted = sigma_batches[batch,:].data.numpy().tolist()
                cell_state = cell_state_batch[batch,:].data.numpy().tolist()
                normalised_residual_per_sensor = [(target_i - prediction_i) / sigma_i for target_i, prediction_i, sigma_i in zip(ground_truth, mu_predicted, sigma_predicted)]
                normalised_residual = sum(normalised_residual_per_sensor) / self.model.input_dim
                    
                data = [id_target[batch].item()] + ground_truth + mu_predicted + sigma_predicted + [normalised_residual] + \
                normalised_residual_per_sensor + cell_state
                batch_results.append(data)

        return batch_results

    def detect_anomaly(self, results_prediction):
        # Drop all empty columns
        results_prediction.drop(results_prediction.columns[results_prediction.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
        for no_sensor in range(1,self.model.input_dim+1):
            sensor_target = results_prediction.iloc[:,no_sensor].values
            sensor_mu = results_prediction.iloc[:,no_sensor+self.model.input_dim].values
            sensor_sigma = results_prediction.iloc[:,no_sensor+2*self.model.input_dim].values
            
            # tag sample as an anomaly (1) if sensor value is higher or lower than given confidence band, otherwhise 0 
            anomaly = []
            for target_i, mu_i, sigma_i in zip(sensor_target, sensor_mu, sensor_sigma):
                if target_i < (mu_i - self.no_standard_deviation * sigma_i):
                    anomaly.append(1)
                elif target_i > (mu_i + self.no_standard_deviation * sigma_i):
                    anomaly.append(1)
                else:
                    anomaly.append(0)
            
            results_prediction["Anomaly Sensor_"+str(no_sensor)]= anomaly
        return results_prediction

