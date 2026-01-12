import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.n_inputs + 1)

    def forward(self, input_vec):
        return np.where(np.dot(input_vec, self.weights) > 0, 1, -1)

    def compute_loss(self, inputs, labels):
        predictions = self.forward(inputs)
        mistake_index = (predictions != labels)
        mistake_input = inputs[mistake_index]
        w_norm = np.linalg.norm(self.weights[:self.n_inputs])
        if w_norm < 1e-8:
            return 0
        distances = np.abs(np.dot(mistake_input, self.weights)) / w_norm
        return np.sum(distances)

    def train(self, training_inputs, labels):
        n_samples = training_inputs.shape[0]
        vec = np.hstack((training_inputs, np.ones((n_samples, 1))))
        losses = []
        for epoch in range(self.max_epochs):
            predictions = self.forward(vec)
            mistake_index = (predictions != labels)
            loss = self.compute_loss(vec, labels)
            losses.append(loss)
            if not np.any(mistake_index):
                print(f"模型已经收敛，次数为{epoch}")
                break
            mistake_input = vec[mistake_index]
            mistake_label = labels[mistake_index]
            gradient = np.sum(mistake_label[:, np.newaxis] * mistake_input, axis=0)
            self.weights = self.weights + self.learning_rate * gradient
            print(f"完成{epoch}次训练, 当前损失: {loss:.4f}")
        return losses

def generate_gaussian_data(mean1, cov1, mean2, cov2):
    n_per_class = 100
    class1 = np.random.multivariate_normal(mean1, cov1, n_per_class)
    class2 = np.random.multivariate_normal(mean2, cov2, n_per_class)
    labels1 = np.ones(n_per_class)
    labels2 = -np.ones(n_per_class)
    X_train = np.vstack((class1[:80], class2[:80]))
    y_train = np.hstack((labels1[:80], labels2[:80]))
    X_test = np.vstack((class1[80:], class2[80:]))
    y_test = np.hstack((labels1[80:], labels2[80:]))
    return X_train, y_train, X_test, y_test

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def plot_decision_boundary(perceptron, X_train, y_train, X_test, y_test, title="Decision Boundary"):
    plt.figure(figsize=(8, 6))

    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                marker='o', s=60, facecolors='blue', edgecolors='k', label='Train +1')
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1],
                marker='s', s=60, facecolors='red', edgecolors='k', label='Train -1')

    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
                marker='o', s=160, facecolors='none', edgecolors='blue', linewidths=1.5, label='Test +1')
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1],
                marker='s', s=160, facecolors='none', edgecolors='red', linewidths=1.5, label='Test -1')

    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    plot_x = np.linspace(x_min, x_max, 300)
    w1, w2, b = perceptron.weights
    if abs(w2) > 1e-8:
        plot_y = (-w1 * plot_x - b) / w2
        plt.plot(plot_x, plot_y, 'k--', label='Decision Boundary')

    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_curve(losses, title="Loss Curve"):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)), losses[1:], marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Sum of distances of misclassified points)")
    plt.grid(True)
    plt.show()

def run_experiment(title, mean1, cov1, mean2, cov2):
    print(f"\n{'=' * 20}\nRunning Experiment: {title}\n{'=' * 20}")
    X_train, y_train, X_test, y_test = generate_gaussian_data(mean1, cov1, mean2, cov2)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    p = Perceptron(n_inputs=2, max_epochs=100, learning_rate=0.01)
    losses = p.train(X_train, y_train)
    print(f"最终权重 (w1, w2, bias): {p.weights}")
    X_test_b = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    predictions = p.forward(X_test_b)
    test_acc = accuracy(y_test, predictions)
    print(f"测试集准确率: {test_acc * 100:.2f}%")
    plot_decision_boundary(p, X_train, y_train, X_test, y_test, title=title)
    plot_loss_curve(losses, title=f"{title} - Loss Curve")

if __name__ == "__main__":
    mean_separable = np.array([-2, 2])
    cov_separable = np.array([[1, 0], [0, 1]])
    run_experiment(
        title="Task 1.3: Linearly Separable Data",
        mean1=mean_separable,
        cov1=cov_separable,
        mean2=-mean_separable,
        cov2=cov_separable
    )

    mean_close1 = np.array([0.5, 0.5])
    mean_close2 = np.array([-0.5, -0.5])
    cov_close = np.array([[1, 0.5], [0.5, 1]])
    run_experiment(
        title="Task 1.4a: Means Too Close (Linearly Inseparable)",
        mean1=mean_close1,
        cov1=cov_close,
        mean2=mean_close2,
        cov2=cov_close
    )

    mean_high_var = np.array([3, 3])
    cov_high_var = np.array([[4, 2], [2, 4]])
    run_experiment(
        title="Task 1.4b: Variance Too High (Linearly Inseparable)",
        mean1=mean_high_var,
        cov1=cov_high_var,
        mean2=-mean_high_var,
        cov2=cov_high_var
    )
