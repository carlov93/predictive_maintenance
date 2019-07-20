import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np

# own Modules 
from models import LstmMse

class Trainer(batch_size, input_dim, n_hidden, n_layers, base_lr, max_lr, 
              step_size_up, mode, gamma, criterion, location_model, location_stats, patience, epoch):
    def __init__(self):
        # initialize model, opitimizer, scheduler 
        self.model = LstmMse(batch_size, input_dim, n_hidden, n_layers)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.) 
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr, 
                                                           max_lr, step_size_up, mode, gamma)
        # initialize further variables
        self.criterion = criterion
        self.epoch_training_loss = []
        self.epoch_validation_loss []
        self.lowest_loss = 99
        self.trails = 0
        self.location_model = location_model
        self.location_stats = location_stats
        self.patience = patience
        self.epoch = epoch
    
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
    
    def evaluate(self, data_loader_validation):
        for batch_number, data in enumerate(data_loader_validation):
            input_data, target_data = data
            self.model.eval()
            hidden = self.model.init_hidden()
            output = self.model(input_data, hidden)

            # Calculate loss
            loss = self.criterion(output, target_data)
            self.epoch_validation_loss.append(loss.item())
            mean_epoch_validation_loss = sum(self.epoch_validation_loss) / float(len(self.epoch_validation_loss))
            print("-------- epoch_no. {} finished with eval loss {}--------".format(self.epoch, mean_epoch_validation_loss))
            return mean_epoch_validation_loss
       
    def save_model(self, mean_epoch_validation_loss):
        if mean_epoch_validation_loss < self.lowest_loss:
            self.trials = 0
            self.lowest_loss = mean_epoch_validation_loss
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_epoch_validation_loss
            }, self.location_model)
            print("Epoch {}: best model saved with loss: {}".format(self.epoch, mean_epoch_validation_loss))
            return True
    
        # Else: Increase trails by one and start new epoch as long as not too many epochs 
        # were unsuccessful (controlled by patience)
        else:
            self.trials += 1
            if self.trials >= self.patience :
                print(f'Early stopping on epoch {epoch}')
                return False
            return True
    
    def save_statistic(self, hist_loss):
        df = pd.DataFrame(hist_loss)
        df.to_csv(self.location_stats, sep=";", index=False)
        