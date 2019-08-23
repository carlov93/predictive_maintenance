import torch
import torch.nn as nn
import csv

class LstmMse_LatentSpace(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, 
                 dropout_rate, n_hidden_fc_prediction, n_hidden_fc_ls_analysis):
        super(LstmMultiTaskLearning, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_hidden_fc_prediction = n_hidden_fc_prediction
        self.n_hidden_fc_ls_analysis = n_hidden_fc_ls_analysis
        self.current_latent_space = None
        
        # define strcture of model
        self.sharedlayer = nn.LSTM(input_size = self.input_dim, 
                                   hidden_size = self.n_hidden_lstm, 
                                   num_layers = self.n_layers, 
                                   batch_first = True, 
                                   dropout = self.dropout_rate)
        
        self.prediction_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_prediction),
                                                nn.Dropout(p=self.dropout_rate),
                                                nn.Tanh(),
                                                nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
                                               )
        
        self.latent_space_analyse_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_ls_analysis),
                                                          nn.Dropout(p=self.dropout_rate),
                                                          nn.Tanh(),
                                                          AnalysisLayer(),
                                                          nn.Linear(self.n_hidden_fc_ls_analysis, self.input_dim)
                                                          )        

    def forward(self, input_data, hidden):
        # Forward propagate LSTM
        # LSTM in Pytorch return two results: the first one usually called output 
        # and the second one (hidden_state, cell_state). 
        lstm_out, (hidden_state, cell_state)= self.sharedlayer(input_data, hidden)
        
        # LSTM returns as output all the hidden_states for all the timesteps (seq), 
        # in other words all of the hidden states throughout the sequence.
        # Thus we have to select the output from the last sequence (last hidden state of sequence).
        # Length of input data can varry 
        length_seq = input_data.size()[1]
        last_out = lstm_out[:,length_seq-1,:]
        
        # Define forward pass through both sub-networks
        prediction = self.prediction_network(last_out)
        _ = self.latent_space_analyse_network(last_out)
        
        # Save latent space
        self.current_latent_space = latent_space
        
        return prediction, _
        
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]