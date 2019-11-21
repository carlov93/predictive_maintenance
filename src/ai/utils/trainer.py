import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins
from random import randint

class Trainer():
    def __init__(self, model, optimizer, scheduler, scheduler_active, criterion, 
                 patience, location_model, location_stats):
        self.model = model
        # lr=1. because of scheduler (1*learning_rate_schedular)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_active = scheduler_active
        # initialize further variables
        self.criterion = criterion
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
        self.lowest_loss = 99
        self.trials = 0
        self.fold = "final_model_"
        self.patience = patience
        self.location_model = location_model
        self.location_stats = location_stats
    
    def train(self, data_loader_training):
        for batch_number, (input_data, target_data) in enumerate(data_loader_training):
            # The LSTM has to be reinitialised, otherwise the LSTM will treat a new batch 
            # as a continuation of a sequence. When batches of data are independent sequences, 
            # then you should reinitialise the hidden state before each batch. 
            # But if your data is made up of really long sequences and you cut it up into batches 
            # making sure that each batch follows on from the previous batch, then in that case 
            # you wouldn’t reinitialise the hidden state before each batch.
            # In the current workflow of class DataProvoider independent sequences are returned. 
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

            # Update LR if scheduler is active 
            if self.scheduler_active:
                self.scheduler.step()
            
        # Return mean of loss over all training iterations
        return sum(self.epoch_training_loss) / float(len(self.epoch_training_loss))
    
    def evaluate(self, data_loader_validation, hist_loss, epoch):
        for batch_number, data in enumerate(data_loader_validation):
            with torch.no_grad():
                input_data, target_data = data
                self.model.eval()
                hidden = self.model.init_hidden()
                output = self.model(input_data, hidden)

                # Calculate loss
                loss = self.criterion(output, target_data)
                self.epoch_validation_loss.append(loss.item())
            
        # Return mean of loss over all validation iterations
        return sum(self.epoch_validation_loss) / float(len(self.epoch_validation_loss))
            
    def cache_history_training(self, hist_loss, epoch, mean_epoch_training_loss, mean_epoch_validation_loss):
        # Save training and validation loss to history
        history = {'epoch': epoch, 'training': mean_epoch_training_loss, 'validation': mean_epoch_validation_loss}
        hist_loss.append(history)     
        print("-------- epoch_no. {} finished with eval loss {}--------".format(epoch, mean_epoch_validation_loss))
            
        # Empty list for new epoch 
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
                 
    def save_model(self, epoch, mean_epoch_validation_loss, ID):
        
        path_model = self.location_model+self.fold+"id"+ID
        
        if mean_epoch_validation_loss < self.lowest_loss:
            self.trials = 0
            self.lowest_loss = mean_epoch_validation_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_epoch_validation_loss
            }, path_model)
            print("Epoch {}: best model saved with loss: {}".format(epoch, mean_epoch_validation_loss))
            return True, path_model
    
        # Else: Increase trails by one and start new epoch as long as not too many epochs 
        # were unsuccessful (controlled by patience)
        else:
            self.trials += 1
            if self.trials >= self.patience :
                print("Early stopping on epoch {}".format(epoch))
                return False, path_model
            return True, path_model
                     
class TrainerLatentSpaceAnalyser():
    def __init__(self, model, optimizer, scheduler, scheduler_active, criterion_prediction, 
                 criterion_ls, patience, location_model, location_stats):
        self.model = model
        # lr=1. because of scheduler (1*learning_rate_schedular)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_active = scheduler_active
        # initialize further variables
        self.criterion_prediction = criterion_prediction
        self.criterion_ls = criterion_ls
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
            prediction, _ = self.model(input_data, hidden)
            
            # Calculate loss
            l1 = self.criterion_prediction(prediction, target_data)   
            l2 = self.criterion_ls(_, target_data)
            loss =  (l1 + l2) / 2
            self.epoch_training_loss.append(loss.item())

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Update LR if scheduler is active 
            if self.scheduler_active:
                self.scheduler.step()
            
        # Return mean of loss over all training iterations
        return sum(self.epoch_training_loss) / float(len(self.epoch_training_loss))
    
    def evaluate(self, data_loader_validation, hist_loss, epoch):
        for batch_number, data in enumerate(data_loader_validation):
            with torch.no_grad():
                input_data, target_data = data
                self.model.eval()
                hidden = self.model.init_hidden()

                # Forward propagation
                prediction, _ = self.model(input_data, hidden)

                # Calculate loss
                l1 = self.criterion_prediction(prediction, target_data)   
                l2 = self.criterion_ls(_, target_data)
                loss =  (l1 + l2)/2
                self.epoch_validation_loss.append(loss.item())
            
        # Return mean of loss over all validation iterations
        return sum(self.epoch_validation_loss) / float(len(self.epoch_validation_loss))
            
    def cache_history_training(self, hist_loss, epoch, mean_epoch_training_loss, mean_epoch_validation_loss):
        # Save training and validation loss to history
        print("-------- epoch_no. {} finished with eval loss {}--------".format(epoch, mean_epoch_validation_loss))
        hist_loss.append(history) 
        return {'epoch': epoch, 'training': mean_epoch_training_loss, 'validation': mean_epoch_validation_loss}
            
        # Empty list for new epoch 
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
        
    def save_model(self, epoch, mean_epoch_validation_loss, input_size, 
                   n_lstm_layer, n_hidden_lstm, n_hidden_fc_pred, seq_size, n_hidden_fc_ls):
        if mean_epoch_validation_loss < self.lowest_loss:
            self.trials = 0
            self.lowest_loss = mean_epoch_validation_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': mean_epoch_validation_loss
            }, self.location_model+"_InputSize"+str(input_size)+"_LayerLstm"+
                str(n_lstm_layer)+"_HiddenLstm"+str(n_hidden_lstm)+"_HiddenFc_pred"+
                str(n_hidden_fc_pred)+"_HiddenFc_ls"+str(n_hidden_fc_ls)+"_Seq"+str(seq_size)+".pt")
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
    
    def save_statistic(self, hist_loss, sequenze_size, n_lstm_layer, n_hidden_lstm, n_hidden_fc, time):
        with open(self.location_stats, 'a') as file:
            file.write("\n"+str(round(min(hist_loss),2))+","+str(sequenze_size)+","+str(n_lstm_layer)+","+ \
                       str(n_hidden_lstm)+","+str(n_hidden_fc)+","+str(round(time,1))) 