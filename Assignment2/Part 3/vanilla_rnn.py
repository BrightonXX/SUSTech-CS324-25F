from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.linear_input = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_output = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        current_batch_size = x.size(0)
        seq_length = x.size(1)
        

        h = torch.zeros(current_batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_length):
            x_t = x[:, t, :]
            
            pre_activation = self.linear_input(x_t) + self.linear_hidden(h)
            h = self.tanh(pre_activation)
            
        output = self.linear_output(h)
        
        return output