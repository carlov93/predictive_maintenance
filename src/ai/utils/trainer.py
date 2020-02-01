import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from random import randint


class Trainer:
    """
    This class implements methods for training, evaluating and
    saving a ML model.
    """
    def __init__(self, model, optimizer, scheduler, scheduler_active, criterion, 
                 patience, location_model):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_active = scheduler_active
        self.criterion = criterion
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
        self.lowest_val_loss = 99
        self.lowest_train_loss = 99
        self.trials = 0
        self.patience = patience
        self.location_model = location_model
    
    def train(self, data_loader_training):
        """
        This method implements the model's training of one epoch. It consist of a forward and
        backward pass through the network. When batches of data are independent sequences,
        we have to re-initialise the hidden and cell state before each batch.
        And we have to zero out the gradient after each bach, else they will accumulate.
        :param data_loader_training: DataLoader class provided by Pytorch.
        :return: Mean training loss of epoch.
        """
        for batch_number, (input_data, target_data) in enumerate(data_loader_training):
            self.model.train()
            hidden = self.model.init_hidden()
            self.optimizer.zero_grad()

            output = self.model(input_data, hidden)
            loss = self.criterion(output, target_data)
            self.epoch_training_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()
            if self.scheduler_active:
                self.scheduler.step()

        return sum(self.epoch_training_loss) / float(len(self.epoch_training_loss))
    
    def evaluate(self, data_loader_validation):
        """
        This method takes the validation data and performs the forward pass through the model.
        The loss is calculated for each batch.
        :param data_loader_validation: DataLoader class provided by Pytorch.
        :return: Mean of loss over all validation iterations.
        """
        for batch_number, data in enumerate(data_loader_validation):
            with torch.no_grad():
                input_data, target_data = data
                self.model.eval()
                hidden = self.model.init_hidden()
                output = self.model(input_data, hidden)
                loss = self.criterion(output, target_data)
                self.epoch_validation_loss.append(loss.item())

        return sum(self.epoch_validation_loss) / float(len(self.epoch_validation_loss))
            
    def cache_history_training(self, epoch, mean_epoch_training_loss, mean_epoch_validation_loss):
        """
        This method prints the training metrics of the current epoch and
        clears the list for the next epoch.
        :param epoch: Number of epoch.
        :param mean_epoch_training_loss: Mean of current epoch's training loss.
        :param mean_epoch_validation_loss: Mean of current epoch's validation loss.
        :return:
        """
        f"-------- epoch_no. {epoch} finished with train loss {mean_epoch_training_loss}--------"
        f"-------- epoch_no. {epoch} finished with eval loss {mean_epoch_validation_loss}--------"
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
                 
    def save_model(self, epoch, mean_epoch_validation_loss, ID):
        """
        This method saves the current model, if it performs better on the validation data than
        the last saved model.
        If the performance is worse, the counter is updated. The method returns False if too many epochs
        were unsuccessful (controlled by patience).
        :param epoch: Number of epoch.
        :param mean_epoch_validation_loss: Mean of current epoch's validation loss.
        :param ID: Model's ID.
        :return: Boolean and path where the model should be saved.
        """
        path_model = self.location_model+"id"+ID
        
        if mean_epoch_validation_loss < self.lowest_val_loss:
            self.trials = 0
            self.lowest_val_loss = mean_epoch_validation_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_epoch_validation_loss
            }, path_model)
            print(f"Epoch {epoch}: best model saved with loss: {mean_epoch_validation_loss}")
            return True, path_model
        else:
            self.trials += 1
            if self.trials >= self.patience:
                print(f"Early stopping on epoch {epoch}")
                return False, path_model
            return True, path_model
