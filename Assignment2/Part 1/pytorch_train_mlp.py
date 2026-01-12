from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from torch import nn
from torch import optim
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from pytorch_mlp import MLP
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 32

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    _, predicted_classes = torch.max(predictions, 1)
    correct = (predicted_classes == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def train(data, 
    dnn_hidden_units=DNN_HIDDEN_UNITS_DEFAULT, 
    learning_rate=LEARNING_RATE_DEFAULT, 
    max_steps=MAX_EPOCHS_DEFAULT, 
    eval_freq=EVAL_FREQ_DEFAULT,
    batch_size=BATCH_SIZE_DEFAULT):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    X_train, y_train, X_test, y_test = data

    if y_train.ndim == 2 and y_train.shape[1] > 1:
        y_train_indices = np.argmax(y_train, axis=1)
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_train_indices = y_train
        y_test_indices = y_test

    X_train_t = torch.from_numpy(X_train).float()
    X_test_t = torch.from_numpy(X_test).float()
    
    y_train_t = torch.from_numpy(y_train_indices).long()
    y_test_t = torch.from_numpy(y_test_indices).long()

    n_inputs = X_train.shape[1]
    # 既然我们已经转换成了 Indices，类别的数量就是 One-Hot 的宽度，或者 max index + 1
    n_classes = y_train.shape[1] if y_train.ndim == 2 else len(np.unique(y_train))
    
    hidden_layers = [int(x) for x in dnn_hidden_units.split(',')]
    
    model = MLP(n_inputs, hidden_layers, n_classes)
    
    # 定义 Loss 和 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 训练循环
    steps_log = []
    train_acc_log = []
    test_acc_log = []
    loss_log = []
    
    N = X_train.shape[0]
    
    # 处理 Batch Size 参数
    if batch_size == 'full' or isinstance(batch_size, str):
        curr_bs = N
        print("PyTorch: Full batch gradient descent")
    else:
        curr_bs = int(batch_size)
        print(f"PyTorch: Mini-batch gradient descent with batch size {curr_bs}")

    for epoch in range(max_steps):
        model.train()
        
        permutation = torch.randperm(N)
        X_train_shuffled = X_train_t[permutation]
        y_train_shuffled = y_train_t[permutation]
        
        for i in range(0, N, curr_bs):
            x_batch = X_train_shuffled[i:i+curr_bs]
            y_batch = y_train_shuffled[i:i+curr_bs]
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad() # 清空梯度
            loss.backward()       # 反向传播
            optimizer.step()      # 更新参数
            
        # Evaluation
        if epoch % eval_freq == 0 or epoch == max_steps - 1:
            model.eval()
            with torch.no_grad(): 
                # Train Accuracy
                train_logits = model(X_train_t)
                train_acc = accuracy(train_logits, y_train_t)
                test_logits = model(X_test_t)
                test_loss = criterion(test_logits, y_test_t).item()
                test_acc = accuracy(test_logits, y_test_t)
            
            steps_log.append(epoch)
            train_acc_log.append(train_acc)
            test_acc_log.append(test_acc)
            loss_log.append(loss.item()) # 记录的是最后一个 batch 的 loss，或者你可以记录 test_loss
            
            print(f"[PyTorch] Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}")

    print("PyTorch Training complete!")
    return steps_log, train_acc_log, test_acc_log, loss_log, model

def main():
    """
    Main function
    """
    # 为了让脚本能独立运行测试，我们在 main 里生成数据
    # 这样不影响 train 函数被 Notebook 调用
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    # 模拟 One-Hot，因为 train 函数里兼容了 One-Hot 处理
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    data = (X_train, y_train, X_test, y_test)
    
    train(data, FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()