import pandas as pd
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import torch

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