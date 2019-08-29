import torch
import torch.nn as nn
import csv

class AnalysisLayer(nn.Module):
    def __init__(self):
        super(AnalysisLayer, self).__init__()
    
    def forward(self, x):
        global latent_space
        latent_space = x.detach()
        return x

class LstmMse_LatentSpace(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, 
                 dropout_rate_fc, dropout_rate_lstm, n_hidden_fc_prediction, n_hidden_fc_ls_analysis):
        super(LstmMse_LatentSpace, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc_prediction = n_hidden_fc_prediction
        self.n_hidden_fc_ls_analysis = n_hidden_fc_ls_analysis
        self.current_latent_space = None
        
        # define strcture of model
        self.sharedlayer = nn.LSTM(input_size = self.input_dim, 
                                   hidden_size = self.n_hidden_lstm, 
                                   num_layers = self.n_layers, 
                                   batch_first = True, 
                                   dropout = self.dropout_rate_lstm)
        
        self.prediction_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_prediction),
                                                nn.Dropout(p=self.dropout_rate_fc),
                                                nn.Tanh(),
                                                nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
                                               )
        
        self.latent_space_analyse_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_ls_analysis),
                                                          nn.Dropout(p=self.dropout_rate_fc),
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

    
class LstmMle_LatentSpace(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, 
                 dropout_rate_fc, dropout_rate_lstm, n_hidden_fc_prediction, n_hidden_fc_ls_analysis, K):
        super(LstmMle_LatentSpace, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc_prediction = n_hidden_fc_prediction
        self.n_hidden_fc_ls_analysis = n_hidden_fc_ls_analysis
        self.current_latent_space = None
        self.K = K
        
        # define strcture of model
        self.sharedlayer = nn.LSTM(input_size = self.input_dim, 
                                   hidden_size = self.n_hidden_lstm, 
                                   num_layers = self.n_layers, 
                                   batch_first = True, 
                                   dropout = self.dropout_rate_lstm)
        
        # define structure of sub network for prediction purpose
        self.p_fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_prediction)
        self.p_dropout = nn.Dropout(p=self.dropout_rate_fc)
        self.p_fc_y_hat = nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
        self.p_fc_tau = nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
        
        # define structure of sub network for latent space analysis
        self.ls_fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_ls_analysis)
        self.ls_dropout = nn.Dropout(p=self.dropout_rate_fc)
        self.ls_analysis = AnalysisLayer(),
        self.ls_fc_y_hat = nn.Linear(self.n_hidden_fc_ls_analysis, self.input_dim)
        self.ls_fc_tau = nn.Linear(self.n_hidden_fc_ls_analysis, self.input_dim)    

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
        
        # Forward pass through sub network for prediction purpose
        p_out = self.p_fc1(last_out)
        p_out = self.p_dropout(p_out)
        p_out = torch.tanh(p_out)
        p_y_hat = self.p_fc_y_hat(p_out)
        p_tau = self.p_fc_tau(p_out)
    
        # Forward pass through sub network for latent space analysis
        ls_out = self.fc1(last_out)
        ls_out = self.dropout(out)
        ls_out = torch.tanh(out)
        ls_out = self.ls_analysis(ls_out)
        ls_y_hat = self.fc_y_hat(out)
        ls_tau = self.fc_tau(out)
        _ = [ls_y_hat, ls_tau * self.K]
        
        # Save latent space
        self.current_latent_space = latent_space
        
        return [p_y_hat, p_tau * self.K], _
        
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]

class LstmMle_LatentSpace_old(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, 
                 dropout_rate_fc, dropout_rate_lstm, n_hidden_fc_prediction, n_hidden_fc_ls_analysis, K):
        super(LstmMle_LatentSpace, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc_prediction = n_hidden_fc_prediction
        self.n_hidden_fc_ls_analysis = n_hidden_fc_ls_analysis
        self.current_latent_space = None
        self.K = K
        
        # define strcture of model
        self.sharedlayer = nn.LSTM(input_size = self.input_dim, 
                                   hidden_size = self.n_hidden_lstm, 
                                   num_layers = self.n_layers, 
                                   batch_first = True, 
                                   dropout = self.dropout_rate_lstm)
        
        # define structure of sub network for prediction purpose
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_prediction)
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        self.fc_y_hat = nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
        self.fc_tau = nn.Linear(self.n_hidden_fc_prediction, self.input_dim)
        
        # define structure of sub network for latent space analysis
        self.latent_space_analyse_network = nn.Sequential(nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_ls_analysis),
                                                          nn.Dropout(p=self.dropout_rate_fc),
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
        
        # Forward path through the subsequent fully connected tanh activation 
        # neural network with 2q output channels
        out = self.fc1(last_out)
        out = self.dropout(out)
        out = torch.tanh(out)
        y_hat = self.fc_y_hat(out)
        tau = self.fc_tau(out)
    
        # Forward pass through sub network for latent space analysis
        _ = self.latent_space_analyse_network(last_out)
        # Save latent space
        self.current_latent_space = latent_space
        
        return [y_hat, tau * self.K], _
        
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]