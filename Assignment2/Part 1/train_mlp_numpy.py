import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy
from modules import SoftMax
from modules import Linear
# Default constants
BATCH_SIZE_DEFAULT = 50
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 32

def accuracy(predictions, targets)->float:
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # DONE Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    predict_result = np.argmax(predictions, axis=1)
    target_result = np.argmax(targets, axis=1)
    return np.sum(predict_result == target_result) / predictions.shape[0]

def train(
        data,
        dnn_hidden_units = DNN_HIDDEN_UNITS_DEFAULT,
        learning_rate = LEARNING_RATE_DEFAULT,
        max_steps = MAX_EPOCHS_DEFAULT,
        eval_freq = EVAL_FREQ_DEFAULT,
        batch_size = BATCH_SIZE_DEFAULT,
        momentum = 0,
        leaky = False
    ):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    if isinstance(batch_size, str) and batch_size.lower() == 'full':
        current_batch_size = data[0].shape[0]
        print("Full batch gradient descent")
    else:
        current_batch_size = int(batch_size)
        if (current_batch_size <= 0):
            raise ValueError("batch_size must be a positive integer or 'full'")
        print(f"Mini-batch gradient descent with batch size {current_batch_size}")

    # DONE: Load your data here
    train_x, train_y, test_x, test_y = data
    n_inputs = train_x.shape[1]
    n_classes = train_y.shape[1]

    # DONE: Initialize your MLP model and loss function (CrossEntropy) here
    hidden_layers = [int(x) for x in dnn_hidden_units.split(',')]
    mlp = MLP(n_inputs=n_inputs,n_hidden=hidden_layers,n_classes=n_classes,leaky=leaky)
    softmax_layer = SoftMax()
    loss_func = CrossEntropy()

    steps_log = []
    train_acc_log = []
    test_acc_log = []
    loss_log = []

    num_samples = train_x.shape[0]
    for epoch in range(max_steps):
        permutation = np.random.permutation(num_samples)
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]

        for i in range(0, num_samples, current_batch_size):
            X_batch = shuffled_x[i:i+current_batch_size]
            y_batch = shuffled_y[i:i+current_batch_size]


            logits = mlp.forward(X_batch)
            probs = softmax_layer.forward(logits)
            loss = loss_func.forward(probs, y_batch)

            # 3. Backward pass (compute gradients)
            initial_grad = loss_func.backward()
            mlp.backward(initial_grad)
            
            # 4. Update weights
            for layer in mlp.layers:
                if isinstance(layer, Linear):
                    if momentum == 0:
                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']
                    else:
                        layer.velocity['weight'] = (momentum * layer.velocity['weight'] +
                                                    learning_rate * layer.grads['weight'])
                        layer.params['weight'] -= layer.velocity['weight']
                        layer.velocity['bias'] = (momentum * layer.velocity['bias'] +
                                                  learning_rate * layer.grads['bias'])
                        layer.params['bias'] -= layer.velocity['bias']
        if epoch % eval_freq == 0 or epoch == max_steps - 1:
            # 在完整的测试集上评估
            test_logits = mlp.forward(test_x)
            test_probs = softmax_layer.forward(test_logits)
            test_loss = loss_func.forward(test_probs, test_y)
            test_accuracy = accuracy(test_probs, test_y)

            # 在完整的训练集上评估
            train_logits = mlp.forward(train_x)
            train_probs = softmax_layer.forward(train_logits)
            train_accuracy = accuracy(train_probs, train_y)

            print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            steps_log.append(epoch)
            train_acc_log.append(train_accuracy)
            test_acc_log.append(test_accuracy)
            loss_log.append(loss)

    print("Training complete!")
    return steps_log, train_acc_log, test_acc_log, loss_log, mlp
def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=str, default=BATCH_SIZE_DEFAULT,
                        help='Batch size')
    FLAGS = parser.parse_known_args()[0]
    data = None
    train(data,FLAGS.batch_size,FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
