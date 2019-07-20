import torch
import torch.nn as nn

class LossModuleMse(torch.nn.Module):
    def __init__(self, input_size, batch_size):
        """
        In the constructor we instantiate the module and assign them as
        member variables.
        """
        super(LossModuleMse, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size

    def forward(self, output, target_data):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. 
        """
        # Compute loss
        mean_loss = torch.sum((output - target_data)**2, dim=1) / self.input_size
        mean_loss = torch.sum(mean_loss) / self.batch_size
        return mean_loss

class LossModuleMle(torch.nn.Module):
    def __init__(self, batch_size):
        """
        In the constructor we instantiate the module and assign them as
        member variables.
        """
        super(LossModuleMle, self).__init__()
        self.batch_size = batch_size

    def forward(self, y_hat, sigma, target_data):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. 
        """
        # Compute loss
        term = torch.pow(((target_data-y_hat)/sigma), 2) + 2 * torch.log(sigma)
        loss_batches = torch.sum(input=term, dim=1)
        mean_loss = torch.sum(loss_batches)/self.batch_size
        return mean_loss
    