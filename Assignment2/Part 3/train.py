from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    REAL_INPUT_DIM = 10 
    
    model = VanillaRNN(
        seq_length=config.input_length,
        input_dim=REAL_INPUT_DIM,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
        batch_size=config.batch_size
    ).to(device)

    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    losses = []
    accuracies = []

    print("Start Training...")
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        batch_inputs = batch_inputs.long()

        batch_inputs_onehot = F.one_hot(batch_inputs, num_classes=10).float()
        
        batch_inputs_onehot = batch_inputs_onehot.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        
        logits = model(batch_inputs_onehot)
        
        loss_val = criterion(logits, batch_targets.long())

        loss_val.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        _, predicted = torch.max(logits, 1)
        correct = (predicted == batch_targets).sum().item()
        accuracy = correct / batch_targets.size(0)
        
        losses.append(loss_val.item())
        accuracies.append(accuracy)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss_val.item():.4f}, Accuracy: {accuracy:.4f}")

        if step == config.train_steps:
            break

    print('Done training.')
    return losses, accuracies, model

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence') # 建议先试 5
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)