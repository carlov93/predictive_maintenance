import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# own Modules 
from models import LstmMse

class Trainer():
    def __init__(self, model, optimizer, scheduler, criterion, location_model, location_stats, patience):
        self.model = model
        # lr=1. because of scheduler (1*learning_rate_schedular)
        self.optimizer = optimizer
        self.scheduler = scheduler
        # initialize further variables
        self.criterion = criterion
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
        self.lowest_loss = 99
        self.trails = 0
        self.location_model = location_model
        self.location_stats = location_stats
        self.patience = patience
    
    def train(self, data_loader_training):
        for batch_number, data in enumerate(data_loader_training):
            # The LSTM has to be reinitialised, otherwise the LSTM will treat a new batch 
            # as a continuation of a sequence. When batches of data are independent sequences, 
            # then you should reinitialise the hidden state before each batch. 
            # But if your data is made up of really long sequences and you cut it up into batches 
            # making sure that each batch follows on from the previous batch, then in that case 
            # you wouldn’t reinitialise the hidden state before each batch.
            # In the current workflow of class DataProvoider independent sequences are returned. 
            input_data, target_data = data
            
            self.model.train()
            hidden = self.model.init_hidden()

            # Zero out gradient, else they will accumulate between minibatches
            self.optimizer.zero_grad()

            # Forward propagation
            output = self.model(input_data, hidden)

            # Calculate loss
            loss = self.criterion(output, target_data)
            self.epoch_training_loss.append(loss.item())

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Update LR
            self.scheduler.step()
            # lr_step = self.optimizer.state_dict()["param_groups"][0]["lr"]
    
    def evaluate(self, data_loader_validation, epoch):
        for batch_number, data in enumerate(data_loader_validation):
            input_data, target_data = data
            self.model.eval()
            hidden = self.model.init_hidden()
            output = self.model(input_data, hidden)

            # Calculate loss
            loss = self.criterion(output, target_data)
            self.epoch_validation_loss.append(loss.item())
            mean_epoch_validation_loss = sum(self.epoch_validation_loss) / float(len(self.epoch_validation_loss))
            print("-------- epoch_no. {} finished with eval loss {}--------".format(epoch, mean_epoch_validation_loss))
            return mean_epoch_validation_loss
       
    def save_model(self, mean_epoch_validation_loss, epoch):
        if mean_epoch_validation_loss < self.lowest_loss:
            self.trials = 0
            self.lowest_loss = mean_epoch_validation_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_epoch_validation_loss
            }, self.location_model)
            print("Epoch {}: best model saved with loss: {}".format(epoch, mean_epoch_validation_loss))
            return True
    
        # Else: Increase trails by one and start new epoch as long as not too many epochs 
        # were unsuccessful (controlled by patience)
        else:
            self.trials += 1
            if self.trials >= self.patience :
                print("Early stopping on epoch {}".format(self.epoch))
                return False
            return True
    
    def save_statistic(self, hist_loss):
        df = pd.DataFrame(hist_loss)
        df.to_csv(self.location_stats, sep=";", index=False)
        