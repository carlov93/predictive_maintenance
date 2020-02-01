import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import torch


class DataSetSensors(Dataset):
    """
    Dataset an abstract class representing a dataset. The sensor data inherit Dataset and
    override the following methods.
    """
    def __init__(self, data, timesteps):
        """
        This method transforms time series data into the structure of samples with input sequence
        and target components. For a one-step predictions, the observations at prior time steps,
        so-called lag observations, are used as input and the target is the observation at the
        current time step. Sample 1: [1, 2, 3], [4]; Sample 2: [2, 3, 4], [5].
        :param data: It is a numpy arrary.
        :param timesteps: Sequence length of time series.
        """
        self.data = data
        self.generator = TimeseriesGenerator(data=self.data, targets=self.data, length=timesteps, batch_size=1)

    def __getitem__(self, index):
        """
        This method reads the data from the csv file. It is memory efficient because all rows
        are not stored in the memory at once but read as required.
        """
        x, y = self.generator[index]
        x_torch = torch.from_numpy(x)
        # Dimension 0 with size 1 (created by TimeseriesGenerator) gets removed
        # because DataLoader will add a dimension 0 with size=batch_size.
        x_torch = torch.squeeze(x_torch, 0)
        y_torch = torch.from_numpy(y)
        y_torch = torch.squeeze(y_torch)
        return (x_torch.float(), y_torch.float()) 

    def __len__(self):
        return len(self.generator)
