import torch
import torch.nn as nn

class LossMse(torch.nn.Module):
    def __init__(self, input_size, batch_size):
        super(LossMse, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size

    def forward(self, output, target_data):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. 
        """
        # Compute loss
        loss_batches = torch.sum((output - target_data)**2, dim=1) / self.input_size
        mean_loss = torch.sum(loss_batches) / self.batch_size
        return mean_loss

    def share_per_sensor(self, output, target_data):
        assert self.batch_size==1,"Batch size has to be 1 for this method"
        loss_per_sensor = (output - target_data)**2
        sum_loss = torch.sum((output - target_data)**2, dim=1)
        result = loss_per_sensor/sum_loss.item()
        return result
    
class LossMle(torch.nn.Module):
    def __init__(self, input_size, batch_size):
        """
        In the constructor we instantiate the module and assign them as
        member variables.
        """
        super(LossMle, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size

    def forward(self, input_data, target_data):
        """
        We are minimizing the the negative log likelihood loss function.
        We write σ_t = exp(τ_t) to guarantee σ > 0 and to provide numerical stability in the learning process.
        """    
        y_hat, tau = input_data
        
        # Compute loss
        term = torch.pow((target_data - y_hat) / torch.exp(tau), 2) + 2 * tau
        loss_batches = torch.sum(input=term, dim=1) / self.input_size
        mean_loss = torch.sum(loss_batches)/self.batch_size    
        return mean_loss
