import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import torch

class DataPreperator():
    def __init__(self, path, ignored_features, stake_training_data, first_order_difference=False):
        self.path = path
        self.dataset = self.load_data()
        self.scaler = StandardScaler()
        self.first_order_difference = first_order_difference
        self.ignored_features = ignored_features
        self.stake = stake_training_data
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def scale_data(self, train_data, validation_data):
        # Initialise standard scaler
        self.scaler.fit(train_data)
        # Transform data
        train_scaled = self.scaler.transform(train_data)
        validation_scaled = self.scaler.transform(validation_data)
        return train_scaled, validation_scaled
        
    def drop_features(self):
        for feature in self.ignored_features:
            self.dataset = self.dataset.drop(labels=feature, axis=1)
            
    def first_order_difference(self):
        self.dataset = self.dataset.diff(periods=1)
        self.dataset = self.dataset.dropna()
        
    def provide_statistics(self):
        return self.scaler.mean_, self.scaler.var_
    
    def prepare_data(self):
        self.drop_features()
        if self.first_order_difference:
            self.first_order_difference()
        amount_training_data = round(len(self.dataset)*self.stake)
        train_data = self.dataset.iloc[0:amount_training_data,:]
        validation_data = self.dataset.iloc[amount_training_data:,:]
        train_preprocessed, validation_preporcessed = self.scale_data(train_data, validation_data)
        return train_preprocessed, validation_preporcessed

class DataPreperatorPrediction():
    def __init__(self, path, ignored_features, mean_training_data, var_training_data, first_order_difference=False):
        self.path = path
        self.dataset = self.load_data()
        self.mean_training_data = mean_training_data
        self.var_training_data = var_training_data
        self.first_order_difference = first_order_difference
        self.ignored_features = ignored_features
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def drop_features(self):
        for feature in self.ignored_features:
            self.dataset = self.dataset.drop(labels=feature, axis=1)
            
    def first_order_difference(self):
        self.dataset = self.dataset.diff(periods=1)
        self.dataset = self.dataset.dropna()
    
    def scale_data(self):
        data_numpy = self.dataset.values
        # Transform data for prediction with mean and variance of training data
        train_scaled = np.zeros(shape=(len(data_numpy[:,0]), data_numpy.shape[1]))
        
        # Copy ID of each sample
        train_scaled[:,0]=data_numpy[:,0]
        
        i = 1   # because first feature is ID
        for mean, var in zip(self.mean_training_data, self.var_training_data):
            train_scaled[:,i] = np.subtract(data_numpy[:,i], mean)
            train_scaled[:,i] = np.divide(train_scaled[:,i], np.sqrt(var))
            i +=1
        return train_scaled
        
    def prepare_data(self):
        self.drop_features()
        if self.first_order_difference:
            self.first_order_difference()
        preprocessed_data = self.scale_data()
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
