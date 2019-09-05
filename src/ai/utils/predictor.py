import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins

class PredictorMse():
    def __init__(self, model, path_data, columns_to_ignore, threshold_anomaly):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore
        self.threshold_anomaly = threshold_anomaly

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")
        
    def create_column_names_result(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data 
                               if column_name not in self.columns_to_ignore+["ID"]]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data 
                                  if column_name not in self.columns_to_ignore+["ID"]]
        column_names_loss_per_sensor = [column_name+" share of reconstruction error " for column_name in self.column_names_data 
                                        if column_name not in self.columns_to_ignore+["ID"]]
        column_names_residuals= ["residual "+column_name for column_name in self.column_names_data 
                                 if column_name not in self.columns_to_ignore+["ID"]]
        
        column_names = ["ID"] + column_names_target + column_names_predicted + ["reconstruction error"] + column_names_loss_per_sensor + column_names_residuals
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
                residuals = [target_i - prediction_i for target_i, prediction_i in zip(ground_truth, predicted_data)]
                reconstruction_error = sum([(ground_truth_i - predicted_i)**2 for ground_truth_i, predicted_i in zip(ground_truth, predicted_data )]) / self.model.input_dim
                reconstruction_error_per_sensor = [(ground_truth_i - predicted_i)**2/reconstruction_error for ground_truth_i, predicted_i in zip(ground_truth, predicted_data )]

                # Add values to dataframe
                data = [id_target[batch].item()] + ground_truth + predicted_data + [reconstruction_error] + \
                reconstruction_error_per_sensor + residuals
                batch_results.append(data)
                        
        return batch_results
        
    def detect_anomaly(self, results_prediction, smooth_rate):
        # Drop all empty columns
        results_prediction.drop(results_prediction.columns[results_prediction.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
        # smooth loss to 
        smoothed_loss = []
        for i,value in enumerate(results_prediction.loc[:,"loss"]):
            if i==0:
                smoothed_loss.append(value)
            else:
                x = smooth_rate  * value + (1 - smooth_rate) * smoothed_loss[-1]
                smoothed_loss.append(x)
        results_prediction["smoothed_loss"]=smoothed_loss
        
        # tag sample as an anomaly (1) if loss is higher than given threshold, otherwhise 0 
        results_prediction["anomaly"] = np.where(results_prediction["smoothed_loss"]>=self.threshold_anomaly, 1, 0)
        return results_prediction
    

class PredictorMle():
    def __init__(self, model, path_data, columns_to_ignore, threshold_anomaly, no_standard_deviation):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore
        self.threshold_anomaly = threshold_anomaly
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

    def detect_anomaly(self, results_prediction, smooth_rate):
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

class PredictorMseLatentSpaceAnalyser():
    def __init__(self, model, path_data, columns_to_ignore, threshold_anomaly):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore
        self.threshold_anomaly = threshold_anomaly

    def get_column_names_data(self):
        with open(self.path_data, 'r') as f:
            header = f.readline().replace('\n','')
            return header.split(",")

    def create_column_names_result(self):
        column_names_target = [column_name+" target" for column_name in self.column_names_data 
                               if column_name not in self.columns_to_ignore+["ID"]]
        column_names_predicted = [column_name+" predicted" for column_name in self.column_names_data 
                                  if column_name not in self.columns_to_ignore+["ID"]]
        column_names_reconstruction_error_per_sensor = [column_name+" share of reconstruction error " for column_name in self.column_names_data 
                                        if column_name not in self.columns_to_ignore+["ID"]]
        column_names_residuals= ["residual "+column_name for column_name in self.column_names_data 
                                 if column_name not in self.columns_to_ignore+["ID"]]
        column_names_latent_space= ["latent_space_"+str(i) for i in range(self.model.n_hidden_fc_ls_analysis)]
        
        column_names = ["ID"] + column_names_target + column_names_predicted + ["reconstruction error"] + \
                        column_names_reconstruction_error_per_sensor + column_names_residuals + column_names_latent_space
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
            prediction, _ = self.model(input_data, hidden)
            latent_space = self.model.current_latent_space
            
            batch_results= []
            for batch in range(self.model.batch_size):
                # Reshape and Calculate prediction metrics
                predicted_data = prediction[batch,:].data.numpy().tolist()
                ground_truth = target_data[batch,:].data.numpy().tolist()
                residuals = [target_i - prediction_i for target_i, prediction_i in zip(ground_truth, predicted_data)]
                latent_space_np = latent_space[batch,:].data.numpy().tolist()
                reconstruction_error = sum([(ground_truth_i - predicted_i)**2 for ground_truth_i, predicted_i in zip(ground_truth, predicted_data )]) / self.model.input_dim
                reconstruction_error_per_sensor = [(ground_truth_i - predicted_i)**2/reconstruction_error for ground_truth_i, predicted_i in zip(ground_truth, predicted_data )]
                loss = loss_prediction_network[batch].item()
                loss_share_per_sensor_np = loss_share_per_sensor[batch,:].data.numpy().tolist()
                    
                # Add values to dataframe
                data = [id_target[batch].item()] + ground_truth + predicted_data + [reconstruction_error] + reconstruction_error_per_sensor + \
                residuals + latent_space_np
                batch_results.append(data)
                        
        return batch_results

        
    def detect_anomaly(self, results_prediction, smooth_rate):
        # Drop all empty columns
        results_prediction.drop(results_prediction.columns[results_prediction.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
        # smooth loss to 
        smoothed_loss = []
        for i,value in enumerate(results_prediction.loc[:,"loss"]):
            if i==0:
                smoothed_loss.append(value)
            else:
                x = smooth_rate  * value + (1 - smooth_rate) * smoothed_loss[-1]
                smoothed_loss.append(x)
        results_prediction["smoothed_loss"]=smoothed_loss
        
        # tag sample as an anomaly (1) if loss is higher than given threshold, otherwhise 0 
        results_prediction["anomaly"] = np.where(results_prediction["smoothed_loss"]>=self.threshold_anomaly, 1, 0)
        return results_prediction


class PredictorMleLatentSpaceAnalyser():
    def __init__(self, model, path_data, columns_to_ignore, threshold_anomaly, no_standard_deviation):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore
        self.threshold_anomaly = threshold_anomaly
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
        column_names_normalised_residuals= [column_name+" normalised residual" for column_name in self.column_names_data 
                                            if column_name not in self.columns_to_ignore+["ID"]]
        column_names_latent_space= ["latent_space_"+str(i) for i in range(self.model.n_hidden_fc_ls_analysis)]
        
        column_names = ["ID"] + column_names_target + column_names_mu_predicted + column_names_sigma_predicted + \
                        ["mean normalised residual"] +  column_names_normalised_residuals + column_names_latent_space
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
            prediction, _ = self.model(input_data, hidden)
            mu, tau = prediction
            latent_space = self.model.current_latent_space
                
            # Because of the transformation of sigma inside the LossModuleMle (σ_t = exp(τ_t))
            # we have to revert this transformation with exp(tau_i).
            sigma_batches = torch.exp(tau)
                
            # Reshape and Calculate prediction metrics
            batch_results= []
            for batch in range(self.model.batch_size):
                mu_predicted = mu[batch,:].data.numpy().tolist()
                ground_truth = target_data[batch,:].data.numpy().tolist()
                sigma_predicted = sigma_batches[batch,:].data.numpy().tolist()
                latent_space_np = latent_space[batch,:].data.numpy().tolist()
                normalised_residual_per_sensor = [(target_i - prediction_i) / sigma_i for target_i, prediction_i, sigma_i in zip(ground_truth, mu_predicted, sigma_predicted)]
                normalised_residual = sum(normalised_residual_per_sensor) / self.model.input_dim
                    
                data = [id_target[batch].item()] + ground_truth + mu_predicted + sigma_predicted + [normalised_residual] + \
                normalised_residual_per_sensor + latent_space_np
                batch_results.append(data)

        return batch_results
     
    def detect_anomaly(self, results_prediction, smooth_rate):
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

class PredictorMleSpecial():
    def __init__(self, model, path_data, columns_to_ignore, threshold_anomaly, no_standard_deviation):
        self.model = model
        self.path_data = path_data
        self.column_names_data = self.get_column_names_data()
        self.columns_to_ignore = columns_to_ignore
        self.threshold_anomaly = threshold_anomaly
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
        column_names_normalised_residuals= [column_name+" normalised residual" for column_name in self.column_names_data 
                                            if column_name not in self.columns_to_ignore+["ID"]]
        column_names_latent_space_y_hat= ["y_hat latent_space_"+str(i) for i in range(self.model.n_hidden_fc_2)]
        column_names_latent_space_tau= ["ltau atent_space_"+str(i) for i in range(self.model.n_hidden_fc_2)]
        
        column_names = ["ID"] + column_names_target + column_names_mu_predicted + column_names_sigma_predicted + \
                    ["mean normalised residual"] +  column_names_normalised_residuals + column_names_latent_space_y_hat + \
                    column_names_latent_space_tau
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
            mu, tau = self.model(input_data, hidden)
            latent_space_y_hat = self.model.current_latent_space_y_hat
            latent_space_tau = self.model.current_latent_space_tau
                
            # Because of the transformation of sigma inside the LossModuleMle (σ_t = exp(τ_t))
            # we have to revert this transformation with exp(tau_i).
            sigma_batches = torch.exp(tau)
                
            # Reshape and Calculate prediction metrics
            batch_results= []
            for batch in range(self.model.batch_size):
                mu_predicted = mu[batch,:].data.numpy().tolist()
                ground_truth = target_data[batch,:].data.numpy().tolist()
                sigma_predicted = sigma_batches[batch,:].data.numpy().tolist()
                latent_space_y_hat_np = latent_space_y_hat[batch,:].data.numpy().tolist()
                latent_space_tau_np = latent_space_tau[batch,:].data.numpy().tolist()
                normalised_residual_per_sensor = [(target_i - prediction_i) / sigma_i for target_i, prediction_i, sigma_i in zip(ground_truth, mu_predicted, sigma_predicted)]
                normalised_residual = sum(normalised_residual_per_sensor) / self.model.input_dim
                    
                data = [id_target[batch].item()] + ground_truth + mu_predicted + sigma_predicted + [normalised_residual] + \
                normalised_residual_per_sensor + latent_space_y_hat_np + latent_space_tau_np
                batch_results.append(data)

        return batch_results
    
    def detect_anomaly(self, results_prediction, smooth_rate):
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