from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        # Initialization here ...
        # nn.LSTM can not be used
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # bias only need to be added to hidden state
        # input modulation gate
        self.Wgx = nn.Linear(input_dim, hidden_dim,bias=True)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # input gate
        self.Wix = nn.Linear(input_dim, hidden_dim,bias=True)
        self.Wih = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # forget gate
        self.Wfx = nn.Linear(input_dim, hidden_dim,bias=True)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # output gate
        self.Wox = nn.Linear(input_dim, hidden_dim,bias=True)
        self.Woh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.linear_output = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Implementation here ...
        batch_size = x.size(0)
        seq_length = x.size(1)

        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_length):
            x_t = x[:, t, :]
            g_t = self.tanh(self.Wgx(x_t) + self.Wgh(h))
            i_t = self.sigmoid(self.Wix(x_t) + self.Wih(h))
            f_t = self.sigmoid(self.Wfx(x_t) + self.Wfh(h))
            o_t = self.sigmoid(self.Wox(x_t) + self.Woh(h))
            c_t = g_t * i_t + c_t * f_t
            h = self.tanh(c_t) * o_t

        p_t = self.linear_output(h)
        return p_t
    # add more methods here if needed