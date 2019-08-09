import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins

class Predictor():
    def __init__(self, model, criterion, path_data, path_residuals, columns_to_ignore):
        self.model = model
        self.criterion = criterion
        self.path_data = path_data
        self.path_residuals = path_residuals
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
        return ["ID"] + column_names_target + column_names_predicted + ["loss"] + column_names_loss_per_sensor
        
    def create_column_names_residuals(self):
        column_names= ["residual "+column_name for column_name in self.column_names_data if column_name!="timestamp"]
        return ["ID"] + column_names
          
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
                output = self.model(input_data, hidden)

                # Calculate loss
                loss = self.criterion(output, target_data)
                loss_share_per_sensor = self.criterion.share_per_sensor(output, target_data)
                
                # Add values to dataframe 
                output = torch.squeeze(output)
                target_data = torch.squeeze(target_data)
                loss_share_per_sensor = torch.squeeze(loss_share_per_sensor)
                target_data_np = target_data.data.numpy().tolist()
                predicted_data_np = output.data.numpy().tolist()
                loss_share_per_sensor_np = loss_share_per_sensor.data.numpy().tolist()
                data = [id_target] + target_data_np + predicted_data_np + [loss.item()] + loss_share_per_sensor_np
                prediction = prediction.append(pd.Series(data, index=prediction.columns), ignore_index=True)
            print("Finished predicting")                      
        return prediction
                   
    def compute_residuals(self, prediction):
        data = pd.read_csv(self.path_data, sep=",")
        residuals = pd.DataFrame(columns=self.create_column_names_residuals())
        for i in range(1,len(data.columns)):
            residuals[residuals.columns[i]] = prediction[prediction.columns[i]] - prediction[prediction.columns[i+len(data.columns)-1]]  
        # ToDo: residuals["timestamp"]=range(prediction.shape[0])
        return residuals

    def save_residuals(self, residuals_result):
        residuals_result.to_csv(self.path_residuals, sep=";", index=False)