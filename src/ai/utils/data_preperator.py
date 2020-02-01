import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import torch


class DataPreperator:
    """
    This class provides methods for pre-processing the sensor data so it can
    be used for training the LSTM network.
    """
    def __init__(self, path, ignored_features, stake_training_data,
                 features_not_to_scale, first_order_difference=False):
        self.path = path
        self.dataset = self.load_data()
        self.scaler = StandardScaler()
        self.first_order_difference = first_order_difference
        self.ignored_features = ignored_features
        self.features_not_to_scale = features_not_to_scale
        self.stake = stake_training_data
        
    def load_data(self):
        return pd.read_csv(self.path)
    
    def scale_data(self, train_data, val_data):
        """
        This method scales the data by removing the mean and scaling to unit variance.
        If features_not_to_scale is not empty, those features are not scaled.
        """
        if len(self.features_not_to_scale) == 0:
            self.scaler.fit(train_data)
            train_scaled = self.scaler.transform(train_data)
            val_scaled = self.scaler.transform(val_data)
        
        else:
            categorical_features_train = train_data.loc[:, self.features_not_to_scale]
            continous_features_train = train_data.drop(labels=self.features_not_to_scale, axis=1)
            categorical_features_val = val_data.loc[:, self.features_not_to_scale]
            continous_features_val = val_data.drop(labels=self.features_not_to_scale, axis=1)

            self.scaler.fit(continous_features_train)
            continous_train_scaled = self.scaler.transform(continous_features_train)
            continous_val_scaled = self.scaler.transform(continous_features_val)

            # Combine categorical and scaled features 
            train_scaled = np.concatenate((continous_train_scaled,
                                           categorical_features_train), axis=1)
            val_scaled = np.concatenate((continous_val_scaled,
                                         categorical_features_val), axis=1)
        return train_scaled, val_scaled
        
    def drop_features(self):
        self.dataset = self.dataset.drop(labels=self.ignored_features, axis=1)
            
    def first_order_difference(self):
        """
        This method calculates the 1-th order discrete difference along the given axis
        for removing a trend.
        """
        self.dataset = self.dataset.diff(periods=1)
        self.dataset = self.dataset.dropna()
        
    def provide_statistics(self):
        return self.scaler.mean_, self.scaler.var_
    
    def prepare_data(self):
        """
        This function wraps the pre-processing methods and split the data into train
        and validation data.
        :return: Training and val data with dimension [batch, sequence_length, features]
        """
        self.drop_features()
        if self.first_order_difference:
            self.first_order_difference()
        amount_training_data = round(len(self.dataset)*self.stake)
        train_data = self.dataset.iloc[0:amount_training_data, :]
        val_data = self.dataset.iloc[amount_training_data:, :]
        train_preprocessed, val_preporcessed = self.scale_data(train_data, val_data)
        return train_preprocessed, val_preporcessed


class DataPreperatorPrediction:
    """
    This class provides methods for scaling the sensor data accordingly to the mean
    and variance of the training data.
    The first column of the csv file has to be the ID of each sample!
    """
    def __init__(self, path, ignored_features, mean_training_data,
                 var_training_data, first_order_difference=False):
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
        """
        This method calculates the 1-th order discrete difference along the given axis
        for removing a trend.
        """
        self.dataset = self.dataset.diff(periods=1)
        self.dataset = self.dataset.dropna()
    
    def scale_data(self):
        data_numpy = self.dataset.values
        data_scaled = np.zeros(shape=(len(data_numpy[:, 0]), data_numpy.shape[1]))
        
        # Copy ID of each sample
        data_scaled[:, 0] = data_numpy[:, 0]
        
        i = 1   # because first (i=0) feature is ID
        for mean, var in zip(self.mean_training_data, self.var_training_data):
            data_scaled[:, i] = np.subtract(data_numpy[:, i], mean)
            data_scaled[:, i] = np.divide(data_scaled[:, i], np.sqrt(var))
            i += 1
        return data_scaled
        
    def prepare_data(self):
        """
        This function wraps the pre-processing methods.
        :return:
        """
        self.drop_features()
        if self.first_order_difference:
            self.first_order_difference()
        preprocessed_data = self.scale_data()
        return preprocessed_data    
