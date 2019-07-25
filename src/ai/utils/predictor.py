import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins

class Predictor():
    def __init__(self, model, criterion, path_data, path_results, path_residuals):
        self.model = model
        self.criterion = criterion
        self.path_data = path_data
        self.path_prediction_results = path_results
        self.path_residuals = path_residuals
        self.column_names_data = self.get_column_names_data()

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")
        
    def create_column_names_result(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data if column_name!="timestamp"]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data if column_name!="timestamp"]
        return ["timestamp"] + column_names_target + column_names_predicted + ["loss"]
        
    def create_column_names_residuals(self):
        column_names= ["residual "+column_name for column_name in self.column_names_data if column_name!="timestamp"]
        return ["timestamp"] + column_names
          
    def predict(self, data_loader):
        prediction = pd.DataFrame(columns=self.create_column_names_result())
        
        print("Start predicting")
        for batch_number, data in enumerate(data_loader):
            input_data, target_data = data
            hidden = self.model.init_hidden()

            # Forward propagation
            output = self.model(input_data, hidden)

            # Calculate loss
            loss = self.criterion(output, target_data)

            # Add values to dataframe 
            output = torch.squeeze(output)
            target_data = torch.squeeze(target_data)
            target_data_np = target_data.data.numpy().tolist()
            predicted_data_np = output.data.numpy().tolist()
            data = [batch_number] + target_data_np + predicted_data_np + [loss.item()]
            prediction = prediction.append(pd.Series(data, index=prediction.columns), ignore_index=True)
        print("Finished predicting")                      
        return prediction
                   
    def compute_residuals(self, prediction):
        data = pd.read_csv(self.path_data, sep=",")
        residuals = pd.DataFrame(columns=self.create_column_names_residuals())
        for i in range(1,len(data.columns)):
            residuals[residuals.columns[i]] = prediction[prediction.columns[i]] - prediction[prediction.columns[i+len(data.columns)-1]]
        residuals["timestamp"]=data["timestamp"]
        return residuals
        
    def save_prediction(self, prediction_result):
        prediction_result.to_csv(self.path_prediction_results, sep=";", index=False)

    def save_residuals(self, residuals_result):
        residuals_result.to_csv(self.path_residuals, sep=";", index=False)
