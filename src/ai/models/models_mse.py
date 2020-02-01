import torch
import torch.nn as nn


class LstmMse(nn.Module):
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate_lstm, dropout_rate_fc, n_hidden_fc):
        """
        :param batch_size: Number of samples for each batch.
        :param input_dim: Number of dimension of input data.
        :param n_hidden_lstm: Number of hidden units for each LSTM layer.
        :param n_layers: Number of LSTM layer.
        :param dropout_rate_lstm: Percentage of dropout rate for every LSTM layer.
        :param dropout_rate_fc: Percentage of dropout rate for every FC layer.
        :param n_hidden_fc: Number of hidden units for each FC layer.
        """
        super(LstmMse, self).__init__()
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc = n_hidden_fc
        # Definition of layers
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.n_hidden_lstm,
                            num_layers=self.n_layers,
                            batch_first=True,
                            dropout=self.dropout_rate_lstm)
        self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc)
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        self.fc2 = nn.Linear(self.n_hidden_fc, self.input_dim)

    def forward(self, input_data, hidden):
        """
        This method defines the forward pass through neural network.
        :param input_data: Input data with dimension [batch, sequence length, features].
        :param hidden: Contains a tuple of initial hidden state and cell state (h_0, c_0) for each element in the batch.
        :return: Prediction for each dimension (sensor) for x(t+1).
        """
        # LSTM in Pytorch return two results: the first one is single variable 
        # and the second one a tuple of (hidden_state, cell_state).
        lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)

        # LSTM returns all the hidden_states for all the timesteps in variable "lstm_out"
        # Thus we have to select the output from the last sequence (last hidden state of sequence)
        length_seq = input_data.size()[1]
        last_out = lstm_out[:, length_seq - 1, :]

        # Forward path through the subsequent fully connected tanh activation neural network
        out_y_hat = self.fc1(last_out)
        out_y_hat = self.dropout(out_y_hat)
        out_y_hat = torch.tanh(out_y_hat)
        out_y_hat = self.fc2(out_y_hat)
        return out_y_hat

    def init_hidden(self):
        """
        # This method initialize the hidden state as well as the cell state.
        :return: None
        """
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]
    