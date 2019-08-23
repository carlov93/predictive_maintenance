import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import builtins

class Tester():
    def __init__(self, model, criterion, ):
        self.model = model
        self.criterion = criterion
        self.test_loss = []
   
    def evaluate(self, data_loader_testing):
        for batch_number, (input_data, target_data) in enumerate(data_loader_testing):
            with torch.no_grad():
                self.model.eval()
                hidden = self.model.init_hidden()
                output = self.model(input_data, hidden)

                # Calculate loss
                loss = self.criterion(output, target_data)
                self.test_loss.append(loss.item())
        
        # Return mean of loss over all test iterations
        return sum(self.test_loss) / float(len(self.test_loss))
    