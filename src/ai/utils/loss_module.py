import torch
import torch.nn as nn


class LossMse(torch.nn.Module):
    """
    This class implements the standard mean squared error loss function.
    """
    def __init__(self, input_size, batch_size):
        super(LossMse, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size

    def forward(self, output, target_data):
        """
        This method defines how the loss is calculated.
        :param output: It is the model's output with dimension (batch, dimension).
        :param target_data: It is the true sensor value with dimension (batch, dimension).
        :return: Mean loss across all batches.
        """
        # Compute loss
        loss_batches = torch.sum((output - target_data)**2, dim=1) / self.input_size
        mean_loss = torch.sum(loss_batches) / self.batch_size
        return mean_loss


class LossMle(torch.nn.Module):
    """
    This class implements the neg. log-likelihood loss function.
    """
    def __init__(self, input_size, batch_size):
        super(LossMle, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size

    def forward(self, output, target_data):
        """
        This method defines how the loss is calculated.
        By defining sigma_t = exp(tau_t) I guarantee sigma > 0 and provide numerical
        stability in the learning process.
        :param output: It is the model's output with dimension (batch, dimension).
        :param target_data: It is the true sensor value with dimension (batch, dimension).
        :return: Mean loss across all batches.
        """
        y_hat, tau = output
        term = torch.pow((target_data - y_hat) / torch.exp(tau), 2) + 2 * tau
        loss_batches = torch.sum(input=term, dim=1) / self.input_size
        mean_loss = torch.sum(loss_batches)/self.batch_size    
        return mean_loss
