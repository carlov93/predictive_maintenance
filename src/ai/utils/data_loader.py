import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import torch

class DataPreperator():
    def __init__(self, path):
        self.path = path
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def preprocess_data(self, train_data, validation_data):
        # Remove time feature
        train_data = train_data.drop(labels="timestamp", axis=1)
        validation_data = validation_data.drop(labels="timestamp", axis=1)
        # Initialise standard scaler
        scaler = StandardScaler()
        scaler.fit(train_data)
        # Transform data
        train_scaled = scaler.transform(train_data)
        validation_scaled = scaler.transform(validation_data)
        return train_scaled, validation_scaled
        
    def provide_data(self, stake_training_data):
        dataset = self.load_data()
        amount_training_data = round(len(dataset)*stake_training_data)
        train_data = dataset.iloc[0:amount_training_data,:]
        validation_data = dataset.iloc[amount_training_data:,:]
        train_preprocessed, validation_preporcessed = self.preprocess_data(train_data, validation_data)
        
        return train_preprocessed, validation_preporcessed

class DataPreperatorPrediction():
    def __init__(self, path):
        self.path = path
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def preprocess_data(self, train_data):
        # Remove time feature
        data = train_data.drop(labels="timestamp", axis=1).values
        
        # Transform data for prediction with mean and variance of training data
        mean_training_data = [-5.37536613e-02, -2.53111489e-04, -8.82854465e+05, 
                              7.79034183e+02, 1.45531178e+04, 1.37766733e+03, 6.50149764e-01] 
        var_training_data = [1.25303578e-01, 1.16898690e-03, 2.86060835e+06, 1.64515717e+06, 
                             6.85728371e+06, 3.63196175e+05, 8.21463343e-03]
        train_scaled = np.zeros(shape=(len(data[:,0]),7))
        i = 0
        for mean, var in zip(mean_training_data, var_training_data):
            train_scaled[:,i] = np.subtract(data[:,i], mean)
            train_scaled[:,i] = np.divide(train_scaled[:,i], np.sqrt(var))
            i +=1
        return train_scaled
        
    def provide_data(self):
        dataset = self.load_data()
        preprocessed_data = self.preprocess_data(dataset)
        return preprocessed_data    

class DataSet(Dataset):
    def __init__(self, data, timesteps):
        # Data as numpy array is provided
        self.data = data
        # Data generator is initialized, batch_size=1 is indipendent of neural network's batch_size 
        self.generator = TimeseriesGenerator(self.data, self.data, length=timesteps, batch_size=1)

    def __getitem__(self, index):
        x, y = self.generator[index]
        x_torch = torch.from_numpy(x)
        # Dimension 0 with size 1 (created by TimeseriesGenerator because of batch_size=1) gets removed 
        # because DataLoader will add a dimension 0 with size=batch_size as well
        x_torch = torch.squeeze(x_torch) # torch.Size([1, timesteps, 7]) --> torch.Size([timesteps, 7])
        y_torch = torch.from_numpy(y)
        y_torch = torch.squeeze(y_torch)
        return (x_torch.float(), y_torch.float()) 

    def __len__(self):
        return len(self.generator)
