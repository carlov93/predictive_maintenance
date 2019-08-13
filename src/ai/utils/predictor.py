import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins

class PredictorMse():
    def __init__(self, model, criterion, path_data, columns_to_ignore):
        self.model = model
        self.criterion = criterion
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")
        
    def create_column_names_result(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_loss_per_sensor = [column_name+" share of loss " for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_residuals= ["residual "+column_name for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        return ["ID"] + column_names_target + column_names_predicted + ["loss"] + column_names_loss_per_sensor + column_names_residuals
          
    def predict(self, data_loader):
        prediction = pd.DataFrame(columns=self.create_column_names_result())
        self.model.eval()
        with torch.no_grad():
            print("Start predicting.")
            for batch_number, data in enumerate(data_loader):
                input_data, target_data = data
                
                # Store ID of target sample 
                id_target = int(target_data[:,0].item())   #ID must be on first position!
                
                # De-select ID feature in both input_data and target data for inference
                input_data = torch.from_numpy(input_data.numpy()[:,:,1:])   # ID must be on first position!
                target_data = torch.from_numpy(target_data.numpy()[:,1:])   # ID must be on first position!

                # Initilize Hidden and Cell State
                hidden = self.model.init_hidden()

                # Forward propagation
                output = self.model(input_data, hidden)

                # Calculate loss
                loss = self.criterion(output, target_data)
                loss_share_per_sensor = self.criterion.share_per_sensor(output, target_data)
                
                # Reshape and Calculate prediction metrics
                output = torch.squeeze(output)
                predicted_data = output.data.numpy().tolist()
                target_data = torch.squeeze(target_data)
                target_data = target_data.data.numpy().tolist()
                loss_share_per_sensor = torch.squeeze(loss_share_per_sensor)
                loss_share_per_sensor = loss_share_per_sensor.data.numpy().tolist()
                residuals = [target_i - prediction_i for target_i, prediction_i in zip(target_data, predicted_data)]
                
                # Add values to dataframe
                data = [id_target] + target_data + predicted_data + [loss.item()] + loss_share_per_sensor + residuals
                prediction = prediction.append(pd.Series(data, index=prediction.columns), ignore_index=True)
                
                # Print status
                if id_target%5000 == 0:
                    print("Current status: " + str(id_target) + " samples are predicted.")
                
            print("Finished predicting.")                      
        return prediction
    
class PredictorMle():
    def __init__(self, model, criterion, path_data, columns_to_ignore):
        self.model = model
        self.criterion = criterion
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")
        
    def create_column_names_result_mse(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_loss_per_sensor = [column_name+" share of loss " for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        column_names_residuals= ["normalised residual "+column_name for column_name in self.column_names_data if column_name not in self.columns_to_ignore+["ID"]]
        return ["ID"] + column_names_target + column_names_predicted + ["loss"] + column_names_loss_per_sensor + column_names_residuals
           
    def predict(self, data_loader):
        prediction = pd.DataFrame(columns=self.create_column_names_result())
        self.model.eval()
        with torch.no_grad():
            print("Start predicting")
            for batch_number, data in enumerate(data_loader):
                input_data, target_data = data
                
                # Store ID of target sample 
                id_target = int(target_data[:,0].item())   #ID must be on first position!
                
                # De-select ID feature in both input_data and target data for inference
                input_data = torch.from_numpy(input_data.numpy()[:,:,1:])   # ID must be on first position!
                target_data = torch.from_numpy(target_data.numpy()[:,1:])   # ID must be on first position!

                # Initilize Hidden and Cell State
                hidden = self.model.init_hidden()

                # Forward propagation
                y_hat, tau = self.model(input_data, hidden)
                
                # Because of the transformation of sigma inside the LossModuleMle (σ_t = exp(τ_t))
                # we have to revert this transformation with exp(tau_i).
                # ToDo: sigma = torch.exp(tau) ????????
                
                # Calculate loss
                loss = self.criterion(y_hat, target_data)
                loss_share_per_sensor = self.criterion.share_per_sensor(y_hat, target_data)
                
                # Reshape and Calculate prediction metrics
                y_hat = torch.squeeze(y_hat)
                predicted_data = y_hat.data.numpy().tolist()
                sigma = torch.squeeze(sigma)
                sigma = sigma.data.numpy().tolist()
                target_data = torch.squeeze(target_data)
                target_data = target_data.data.numpy().tolist()
                loss_share_per_sensor = torch.squeeze(loss_share_per_sensor)
                loss_share_per_sensor = loss_share_per_sensor.data.numpy().tolist()
                normalised_residuals = [(target_i - prediction_i)/sigma_i for target_i, prediction_i, sigma_i in zip(target_data, predicted_data, sigma)]
                
                # Add values to dataframe 
                data = [id_target] + target_data + predicted_data + [loss.item()] + loss_share_per_sensor + normalised_residuals
                prediction = prediction.append(pd.Series(data, index=prediction.columns), ignore_index=True)
                
                # Print status
                if id_target%5000 == 0:
                    print("Current status: " + str(id_target) + " samples are predicted.")
                    
            print("Finished predicting")                      
        return prediction
    