import torch
import torch.nn as nn

class LstmMle_1(nn.Module):
    """
    Last layer of subsequent fully connected tanh activation neural network is split into two linear layers
    to seperate prediction for y_hat and tau. 
    """
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate_lstm, dropout_rate_fc, n_hidden_fc, K):
        super(LstmMle_1, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc = n_hidden_fc
        self.current_latent_space = None
        self.K = K
        
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.n_hidden_lstm, 
                            num_layers = self.n_layers, 
                            batch_first = True, 
                            dropout = self.dropout_rate_lstm)
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        self.fc_y_hat = nn.Linear(self.n_hidden_fc, self.input_dim)
        self.fc_tau = nn.Linear(self.n_hidden_fc, self.input_dim)
        
    def forward(self, input_data, hidden):
        # Forward propagate LSTM
        # LSTM in Pytorch return two results: the first one usually called output 
        # and the second one (hidden_state, cell_state). 
        lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)

        # LSTM returns as output all the hidden_states for all the timesteps (seq), 
        # in other words all of the hidden states throughout
        # the sequence.
        # Thus we have to select the output from the last sequence (last hidden state of sequence)
        # Length of input data can varry 
        length_seq = input_data.size()[1]
        last_out = lstm_out[:,length_seq-1,:]

        # Forward path through the subsequent fully connected tanh activation 
        # neural network with 2q output channels
        out = self.fc1(last_out)
        out = self.dropout(out)
        out = torch.tanh(out)
        # Store current latent space 
        global latent_space
        self.current_latent_space = out.detach()
        # Continue forward pass
        y_hat = self.fc_y_hat(out)
        tau = self.fc_tau(out)
        return [y_hat, tau * self.K]
    
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]

class LstmMle_2(nn.Module):
    """
    Subsequent fully connected tanh activation neural network is split into two sub-networks.
    One is for predicting y_hat, the other for predicting tau.
    """
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate_lstm, dropout_rate_fc, n_hidden_fc, K):
        super(LstmMle_2, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc = n_hidden_fc
        self.current_latent_space = None
        self.K = K
        
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.n_hidden_lstm, 
                            num_layers = self.n_layers, 
                            batch_first = True, 
                            dropout = self.dropout_rate_lstm)
        self.fc1_y_hat = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.fc1_tau = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout_y_hat = nn.Dropout(p=self.dropout_rate_fc)
        self.dropout_tau = nn.Dropout(p=self.dropout_rate_fc)
        self.fc2_y_hat = nn.Linear(self.n_hidden_fc, self.input_dim)
        self.fc2_tau = nn.Linear(self.n_hidden_fc, self.input_dim)
        
    def forward(self, input_data, hidden):
        # Forward propagate LSTM
        # LSTM in Pytorch return two results: the first one usually called output 
        # and the second one (hidden_state, cell_state). 
        lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)

        # LSTM returns as output all the hidden_states for all the timesteps (seq), 
        # in other words all of the hidden states throughout
        # the sequence.
        # Thus we have to select the output from the last sequence (last hidden state of sequence)
        # Length of input data can varry 
        length_seq = input_data.size()[1]
        last_out = lstm_out[:,length_seq-1,:]

        # Forward path through the subsequent fully connected tanh activation 
        # neural network with 2q output channels
        # Subnetwork for prediction of y_hat
        out_y_hat = self.fc1_y_hat(last_out)
        out_y_hat = self.dropout_y_hat(out_y_hat)
        out_y_hat = torch.tanh(out_y_hat)
        # Store current latent space 
        global latent_space
        self.current_latent_space = out_y_hat.detach()
        # Continue forward pass
        y_hat = self.fc2_y_hat(out_y_hat)
        
        # Subnetwork for prediction of tau
        out_tau = self.fc1_tau(last_out)
        out_tau = self.dropout_tau(out_tau)
        out_tau = torch.tanh(out_tau)
        tau = self.fc2_tau(out)
        
        return [y_hat, tau * self.K]
    
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]