from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        current_dim = n_inputs

        for hidden_units in n_hidden:
            self.layers.append(nn.Linear(current_dim, hidden_units))
            current_dim = hidden_units

        self.output_layer = nn.Linear(current_dim, n_classes)
    

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for layer in self.layers:
            x = layer(x)
            x = nn.ReLU()(x)

        out = self.output_layer(x)
        return out
