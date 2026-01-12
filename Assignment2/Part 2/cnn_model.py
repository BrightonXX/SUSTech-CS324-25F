from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    super(CNN, self).__init__()
    
    # --- Block 1: 32x32 -> 16x16 ---
    # 1. Conv: 3 -> 64
    self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    # 2. MaxPool: 64 -> 64 (Size halves)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    # --- Block 2: 16x16 -> 8x8 ---
    # 3. Conv: 64 -> 128
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    # 4. MaxPool: 128 -> 128 (Size halves)
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    # --- Block 3: 8x8 -> 4x4 ---
    # 5. Conv: 128 -> 256
    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_1 = nn.BatchNorm2d(256)
    # 6. Conv: 256 -> 256
    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_2 = nn.BatchNorm2d(256)
    # 7. MaxPool: 256 -> 256 (Size halves)
    self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # --- Block 4: 4x4 -> 2x2 ---
    # 8. Conv: 256 -> 512
    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_1 = nn.BatchNorm2d(512)
    # 9. Conv: 512 -> 512
    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_2 = nn.BatchNorm2d(512)
    # 10. MaxPool: 512 -> 512 (Size halves)
    self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # --- Block 5: 2x2 -> 1x1 ---
    # 11. Conv: 512 -> 512
    self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5_1 = nn.BatchNorm2d(512)
    # 12. Conv: 512 -> 512
    self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5_2 = nn.BatchNorm2d(512)
    # 13. MaxPool: 512 -> 512 (Size halves)
    self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    # --- Classifier ---
    # Final dimensions: 512 channels * 1 height * 1 width = 512
    self.flat_features = 512 * 1 * 1
    # 14. Linear: 512 -> 10
    self.fc = nn.Linear(self.flat_features, n_classes)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool1(x)
    
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool2(x)
    
    x = F.relu(self.bn3_1(self.conv3_1(x)))
    x = F.relu(self.bn3_2(self.conv3_2(x)))
    x = self.pool3(x)
    
    x = F.relu(self.bn4_1(self.conv4_1(x)))
    x = F.relu(self.bn4_2(self.conv4_2(x)))
    x = self.pool4(x)
    
    x = F.relu(self.bn5_1(self.conv5_1(x)))
    x = F.relu(self.bn5_2(self.conv5_2(x)))
    x = self.pool5(x)
    
    x = x.view(-1, self.flat_features)
    
    out = self.fc(x)
    
    return out