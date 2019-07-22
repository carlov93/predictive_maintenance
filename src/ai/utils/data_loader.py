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
        self.scaler = StandardScaler()
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def preprocess_data(self, train_data, validation_data):
        # Remove time feature
        train_data = train_data.drop(labels="timestamp", axis=1)
        validation_data = validation_data.drop(labels="timestamp", axis=1)
        # Initialise standard scaler
        self.scaler.fit(train_data)
        # Transform data
        train_scaled = self.scaler.transform(train_data)
        validation_scaled = self.scaler.transform(validation_data)
        return train_scaled, validation_scaled
        
    def provide_statistics(self):
        return self.scaler.mean_, self.scaler.var_
    
    def provide_data(self, stake_training_data):
        dataset = self.load_data()
        amount_training_data = round(len(dataset)*stake_training_data)
        train_data = dataset.iloc[0:amount_training_data,:]
        validation_data = dataset.iloc[amount_training_data:,:]
        train_preprocessed, validation_preporcessed = self.preprocess_data(train_data, validation_data)
        
        return train_preprocessed, validation_preporcessed

class DataPreperatorPrediction():
    def __init__(self, path, mean_training_data, var_training_data, input_dim):
        self.path = path
        self.mean_training_data = mean_training_data
        self.var_training_data = var_training_data
        self.input_dim = input_dim
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def preprocess_data(self, train_data):
        # Remove time feature
        data = train_data.drop(labels="timestamp", axis=1).values
        
        # Transform data for prediction with mean and variance of training data
        train_scaled = np.zeros(shape=(len(data[:,0]),self.input_dim))
        i = 0
        for mean, var in zip(self.mean_training_data, self.var_training_data):
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
