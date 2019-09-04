import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataScaler():
    def __init__(self, features_not_to_scale):
        self.scaler = StandardScaler()
        self.features_not_to_scale = features_not_to_scale
        
    def scale_test_dataset(self, test_data, mean_train_data, var_train_data):
        data_numpy = test_data.values
        # Transform data for prediction with mean and variance of training data
        test_scaled = np.zeros(shape=(data_numpy.shape[0], data_numpy.shape[1]))
        
        i = 0
        for mean, var in zip(mean_train_data, var_train_data):
            test_scaled[:,i] = np.subtract(data_numpy[:,i], mean)
            test_scaled[:,i] = np.divide(test_scaled[:,i], np.sqrt(var))
            i +=1
        return test_scaled
    
    def scale_data(self, train_data, validation_data):
        if len(self.features_not_to_scale) == 0:
            # Initialise standard scaler
            self.scaler.fit(train_data)
            # Transform data
            train_scaled = self.scaler.transform(train_data)
            validation_scaled = self.scaler.transform(validation_data)
        
        else:
            # seperate categorical and continous features
            categorical_features_train = train_data.loc[:, self.features_not_to_scale]
            continous_features_train = train_data.drop(labels=self.features_not_to_scale, axis=1)
            categorical_features_validation = validation_data.loc[:, self.features_not_to_scale]
            continous_features_validation = validation_data.drop(labels=self.features_not_to_scale, axis=1)

            # Initialise standard scaler
            self.scaler.fit(continous_features_train)
            # Transform data
            continous_train_scaled = self.scaler.transform(continous_features_train)
            continous_validation_scaled = self.scaler.transform(continous_features_validation)

            # Combine categorical and scaled features 
            train_scaled = np.concatenate((continous_train_scaled, categorical_features_train), axis=1)
            validation_scaled = np.concatenate((continous_validation_scaled, categorical_features_validation), axis=1)
        return train_scaled, validation_scaled
    
    def provide_statistics(self):
        return self.scaler.mean_, self.scaler.var_