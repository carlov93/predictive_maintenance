import torch
import torch.nn as nn

class LstmMle(nn.Module):
    """
    This class implements a LSTM neural network for using a log-likelihood loss function. Depending on the option, the
    network is split into two separate sub-networks in order to predict mean and tau for each input dimension.
    """
    def __init__(self, batch_size, input_dim, n_hidden_lstm, n_layers, dropout_rate_lstm,
                 dropout_rate_fc, n_hidden_fc_1, K, option):
        """
        :param batch_size: Number of samples for each batch.
        :param input_dim: Number of dimension of input data.
        :param n_hidden_lstm: Number of hidden units for each LSTM layer.
        :param n_layers: Number of LSTM layer.
        :param dropout_rate_lstm: Percentage of dropout rate for every LSTM layer.
        :param dropout_rate_fc: Percentage of dropout rate for every FC layer.
        :param n_hidden_fc_1: Number of hidden units for each FC layer.
        :param K: (0 or 1).
        :param option: It defines different architecture types and its corresponding forward pass (1,2 or 3).
        """
        super(LstmMle, self).__init__()
        # Attributes for LSTM Network
        self.input_dim = input_dim
        self.n_hidden_lstm = n_hidden_lstm
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_lstm = dropout_rate_lstm
        self.n_hidden_fc_1 = n_hidden_fc_1
        self.current_cell_state = None
        self.K = K
        self.option = option
        # Definition of NN layer
        # batch_first = True because dataloader creates batches and batch_size is 0. dimension
        if self.option == 1:
            self.lstm = nn.LSTM(input_size=self.input_dim,
                                hidden_size=self.n_hidden_lstm,
                                num_layers=self.n_layers,
                                batch_first=True,
                                dropout=self.dropout_rate_lstm)
            self.fc1 = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_1)
            self.dropout = nn.Dropout(p=self.dropout_rate_fc)
            self.fc_y_hat = nn.Linear(self.n_hidden_fc_1, self.input_dim)
            self.fc_tau = nn.Linear(self.n_hidden_fc_1, self.input_dim)
        elif self.option == 2:
            self.lstm = nn.LSTM(input_size=self.input_dim,
                                hidden_size=self.n_hidden_lstm,
                                num_layers=self.n_layers,
                                batch_first=True,
                                dropout=self.dropout_rate_lstm)
            self.fc1_y_hat = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_1)
            self.fc1_tau = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_1)
            self.dropout_y_hat = nn.Dropout(p=self.dropout_rate_fc)
            self.dropout_tau = nn.Dropout(p=self.dropout_rate_fc)
            self.fc2_y_hat = nn.Linear(self.n_hidden_fc_1, self.input_dim)
            self.fc2_tau = nn.Linear(self.n_hidden_fc_1, self.input_dim)
        elif self.option == 3:
            self.lstm_y_hat = nn.LSTM(input_size=self.input_dim,
                                      hidden_size=self.n_hidden_lstm,
                                      num_layers=self.n_layers,
                                      batch_first=True,
                                      dropout=self.dropout_rate_lstm)
            self.lstm_tau = nn.LSTM(input_size=self.input_dim,
                                    hidden_size=self.n_hidden_lstm,
                                    num_layers=self.n_layers,
                                    batch_first=True,
                                    dropout=self.dropout_rate_lstm)
            self.fc1_y_hat = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_1)
            self.fc1_tau = nn.Linear(self.n_hidden_lstm, self.n_hidden_fc_1)
            self.dropout = nn.Dropout(p=self.dropout_rate_fc)
            self.fc2_y_hat = nn.Linear(self.n_hidden_fc_1, self.input_dim)
            self.fc2_tau = nn.Linear(self.n_hidden_fc_1, self.input_dim)
        
    def forward(self, input_data, hidden):
        """
        This method defines the forward pass through neural network.
        :param input_data: Input data with dimension [batch, sequence length, features].
        :param hidden: Contains a tuple of initial hidden state and cell state (h_0, c_0) for each element in the batch.
        :return: Prediction for each dimension (sensor) for x(t+1).
        """
        if self.option == 1:
            # LSTM returns as output all the hidden_states for all the timesteps (seq).
            # We have to select the output from the last sequence (last hidden state of sequence).
            lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)
            length_seq = input_data.size()[1]
            last_out = lstm_out[:,length_seq-1,:]

            # Only the last layer is split into two linear layer of the prediction of mean and tau. 
            out = self.fc1(last_out)
            out = self.dropout(out)
            out = torch.tanh(out)
            y_hat = self.fc_y_hat(out)
            tau = self.fc_tau(out)

            # Get current cell state
            self.current_cell_state = torch.squeeze(cell_state).detach()

            return [y_hat, tau * self.K]

        if self.option == 2:
            lstm_out, (hidden_state, cell_state) = self.lstm(input_data, hidden)
            length_seq = input_data.size()[1]
            last_out = lstm_out[:, length_seq - 1, :]

            # Subnetwork for prediction of y_hat
            out_y_hat = self.fc1_y_hat(last_out)
            out_y_hat = self.dropout_y_hat(out_y_hat)
            out_y_hat = torch.tanh(out_y_hat)
            y_hat = self.fc2_y_hat(out_y_hat)

            # Subnetwork for prediction of tau
            out_tau = self.fc1_tau(last_out)
            out_tau = self.dropout_tau(out_tau)
            out_tau = torch.tanh(out_tau)
            tau = self.fc2_tau(out_tau)

            # Get current cell state
            self.current_cell_state = torch.squeeze(cell_state).detach()

            return [y_hat, tau * self.K]

        if self.option == 3:
            length_seq = input_data.size()[1]
            # Subnetwork for prediction of y_hat
            lstm_out_y_hat, (hidden_state_y, cell_state_y) = self.lstm_y_hat(input_data, hidden)
            last_out_y_hat = lstm_out_y_hat[:, length_seq - 1, :]
            out_y_hat = self.fc1_y_hat(last_out_y_hat)
            out_y_hat = self.dropout(out_y_hat)
            out_y_hat = torch.tanh(out_y_hat)
            y_hat = self.fc2_y_hat(out_y_hat)

            # Subnetwork for prediction of tau
            lstm_out_tau, (hidden_state_tau, cell_state_tau) = self.lstm_tau(input_data, hidden)
            last_out_tau = lstm_out_tau[:, length_seq - 1, :]
            out_tau = self.fc1_tau(last_out_tau)
            out_tau = self.dropout(out_tau)
            out_tau = torch.tanh(out_tau)
            tau = self.fc2_tau(out_tau)

            # Get current cell state
            self.current_cell_state = torch.squeeze(cell_state_y).detach()

            return [y_hat, tau * self.K]
    
    def init_hidden(self):
        """
        # This method initialize the hidden state as well as the cell state.
        :return: None
        """
        h0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        c0 = torch.zeros(self.n_layers, self.batch_size, self.n_hidden_lstm, requires_grad=False)
        return [t for t in (h0, c0)]
