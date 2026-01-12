from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-3
MAX_EPOCHS_DEFAULT = 30 
BATCH_SIZE_DEFAULT = 64  
EVAL_FREQ_DEFAULT = 1
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data'

FLAGS = None

def train(
    learning_rate=LEARNING_RATE_DEFAULT,
    max_epochs=MAX_EPOCHS_DEFAULT,
    batch_size=BATCH_SIZE_DEFAULT,
    optimizer_type='ADAM',
    use_augmentation=False,
    use_scheduler=False
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: Opt={optimizer_type}, Aug={use_augmentation}, Sch={use_scheduler}, Device={device}")

    # --- 1. Data Preparation with Augmentation ---
    if use_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- 2. Model & Optimizer ---
    model = CNN(n_channels=3, n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 3. Training Loop ---
    train_losses = []
    test_accs = []
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if scheduler:
            scheduler.step()

        # Evaluation per epoch
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        test_accs.append(acc)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

    return train_losses, test_accs