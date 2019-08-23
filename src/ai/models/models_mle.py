import torch
import torch.nn as nn

class LstmMle_1(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate, n_hidden_fc):
        super(LstmMle, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_hidden_fc = n_hidden_fc
        
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.n_hidden_lstm, 
                            num_layers = self.n_layers, 
                            batch_first = True, 
                            dropout = self.dropout_rate)
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout = nn.Dropout(p=self.dropout_rate)
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
        y_hat = self.fc_y_hat(out)
        tau = self.fc_tau(out)
        return [y_hat, tau]
    
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]

class LstmMle_2(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate, n_hidden_fc):
        super(LstmMle, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.n_hidden_fc = n_hidden_fc
        
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size = self.input_dim, 
                            hidden_size = self.n_hidden_lstm, 
                            num_layers = self.n_layers, 
                            batch_first = True, 
                            dropout = self.dropout_rate)
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout = nn.Dropout(p=self.dropout_rate)
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
        y_hat = self.fc_y_hat(out)
        tau = self.fc_tau(out)
        return [y_hat, tau]
    
    def init_hidden(self):
        # This method is for initializing hidden state as well as cell state
        # We need to detach the hidden state to prevent exploding/vanishing gradients
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]