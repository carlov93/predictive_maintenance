import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class Trainer():
    def __init__(self, model, optimizer, scheduler, criterion, patience, location_model, location_stats):
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
        self.patience = patience
        self.location_model = location_model
        self.location_stats = location_stats
    
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
            
        # Return mean of loss over all training iterations
        return sum(self.epoch_training_loss) / float(len(self.epoch_training_loss))
    
    def evaluate(self, data_loader_validation, hist_loss, epoch):
        for batch_number, data in enumerate(data_loader_validation):
            input_data, target_data = data
            self.model.eval()
            hidden = self.model.init_hidden()
            output = self.model(input_data, hidden)

            # Calculate loss
            loss = self.criterion(output, target_data)
            self.epoch_validation_loss.append(loss.item())
            
        # Return mean of loss over all validation iterations
        return sum(self.epoch_validation_loss) / float(len(self.epoch_validation_loss))
           
    def cache_history(self, hist_loss, epoch, mean_epoch_training_loss, mean_epoch_validation_loss):
            # Save training and validation loss to history
            hist_loss.append({'epoch': epoch, 
                              'training': mean_epoch_training_loss, 
                              'validation': mean_epoch_validation_loss})
            print("-------- epoch_no. {} finished with eval loss {}--------".format(epoch, mean_epoch_validation_loss))
            
            # Empty list for new epoch 
            self.epoch_training_loss = []
            self.epoch_validation_loss = []
        
    def save_model(self, mean_epoch_validation_loss, epoch, sequenze_size, n_lstm_layer, n_hidden, stepsize):
        if mean_epoch_validation_loss < self.lowest_loss:
            self.trials = 0
            self.lowest_loss = mean_epoch_validation_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_epoch_validation_loss
            }, self.location_model+"_Seq"+str(sequenze_size)+"_Layer"+
                str(n_lstm_layer)+"_Hidden"+str(n_hidden)+"_Step"+str(stepsize)+".pt")
            print("Epoch {}: best model saved with loss: {}".format(epoch, mean_epoch_validation_loss))
            return True
    
        # Else: Increase trails by one and start new epoch as long as not too many epochs 
        # were unsuccessful (controlled by patience)
        else:
            self.trials += 1
            if self.trials >= self.patience :
                print("Early stopping on epoch {}".format(epoch))
                return False
            return True
    
    def save_statistic(self, hist_loss, sequenze_size, n_lstm_layer, n_hidden, stepsize, time):
        df = pd.DataFrame(hist_loss)
        df.to_csv(self.location_stats+"_Loss"+str(round(self.lowest_loss,2))+"_Seq"+str(sequenze_size)+"_Layer"+
                str(n_lstm_layer)+"_Hidden"+str(n_hidden)+"_Step"+str(stepsize)+"_Time"+str(round(time,0))+".csv",
                  sep=";", index=False)
        