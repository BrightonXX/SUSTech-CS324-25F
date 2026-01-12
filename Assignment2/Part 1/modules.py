import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer.
        DONE: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        weight = np.random.randn(in_features,out_features) * 0.1
        bias = np.zeros(out_features)

        self.params = {'weight': weight, 'bias': bias}
        self.grads = {'weight': None, 'bias': None}
        self.velocity = {'weight': np.zeros_like(weight), 'bias': np.zeros_like(bias)}
        self.x = None

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        DONE: Implement the forward pass.
        """
        self.x = x
        W = self.params['weight']
        b = self.params['bias']
        out = np.dot(x,W) + b
        return out

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        W = self.params['weight']
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        db = np.sum(dout, axis=0)
        self.grads['weight'] = dW
        self.grads['bias'] = db
        return dx

class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        DONE: Implement the forward pass.
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        DONE: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx

class SoftMax(object):
    def __init__(self):
        self.prob = None

    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        DONE: Implement the forward pass using the Max Trick for numerical stability.
        """
        max_x = np.max(x, axis=1, keepdims=True)
        shifted_x = x - max_x
        exped_x = np.exp(shifted_x)
        sum = np.sum(exped_x, axis=1, keepdims=True)
        self.prob = exped_x/sum

        return self.prob

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        DONE Keep this in mind when implementing CrossEntropy's backward method.
        THIS CALCULATION IS NOT DONE AT HERE!!
        """
        return dout

class CrossEntropy(object):
    def __init__(self):
        self.prob = None
        self.y = None

    def forward(self, prob, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        DONE: Implement the forward pass.
        """
        self.y = y
        self.prob = prob
        N = self.prob.shape[0]
        epsilon = 1e-9

        loss = -np.sum(self.y * np.log(self.prob + epsilon))/N
        return loss

    def backward(self, x=None, y=None):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        DONE: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        N = self.prob.shape[0]

        y = self.y

        dx = (self.prob - y) / N

        return dx


class LeakyReLU(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.where(x > 0, x, self.alpha * x)
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] *= self.alpha

        return dx