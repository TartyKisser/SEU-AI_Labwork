import numpy as np
import matplotlib.pyplot as plt
import os
import struct


def mnist_transform(images):
    return images.astype(np.float32) / 255.0


def plot_history(*args, **kwargs):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    if len(args) > 0 and len(kwargs) > 0:
        raise RuntimeError(
            "cannot specify position args and keyword args at the same time")

    data = zip([""] * len(args), args) if len(args) > 0 else kwargs.items()
    for label, his in data:
        ax1.plot([x['train_loss'] for x in his], label=label + ' train loss')
        ax1.plot([x['test_loss'] for x in his], label=label + ' test loss')
        ax2.plot([x['test_acc'] for x in his], label=label + ' accuracy')
    ax2.set_ylabel("Accuracy")
    ax2.set_title('Accuracy on test set vs Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.legend()
    ax2.legend()
    fig.supxlabel("Epochs")
    fig.tight_layout()
    plt.show()


def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 784)
    return images


def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


import numpy as np


class MyDataLoader:
    def __init__(self, data, batch_size=1, shuffle=False):
        """
        Initialize the data loader with a dataset, batch size, and shuffle option.

        Parameters:
            data (tuple): Tuple containing two elements, the images and their corresponding labels.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the indices of the samples.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image, self.label = data
        self.num_samples = len(self.image)
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        """
        Reset the current index and return the iterator.

        Returns:
            self: The instance itself.
        """
        self.current_idx = 0
        return self

    def __next__(self):
        """
        Return the next batch of images and labels.

        Raises:
            StopIteration: If all batches have been fetched.

        Returns:
            tuple: A batch of images and their corresponding labels.
        """
        if self.current_idx >= self.num_samples:
            raise StopIteration

        end_idx = self.current_idx + self.batch_size
        indices = self.indices[self.current_idx:end_idx]
        images = self.image[indices]
        labels = self.label[indices]
        self.current_idx += self.batch_size
        return images, labels


def one_hot(y, num_classes):
    """
    Convert an array of numerical labels to one-hot encoded format.

    Parameters:
        y (int, list, or np.ndarray): The label or array of labels to be one-hot encoded.
        num_classes (int): The total number of classes.

    Returns:
        np.ndarray: One-hot encoded array for the given labels.

    Raises:
        ValueError: If `num_classes` is less than the maximum label in `y`.
        TypeError: If `y` is not of a type that can be converted to an np.ndarray.
    """
    if not isinstance(y, (int, list, np.ndarray)):
        raise TypeError("Labels must be an int, list, or np.ndarray.")

    y_array = np.array(y, dtype=int)
    if np.max(y_array) >= num_classes:
        raise ValueError("Maximum label in `y` should be less than `num_classes`.")

    one_hot_encoded = np.eye(num_classes, dtype=float)[y_array]
    return one_hot_encoded


def relu(x):
    """
    Apply the ReLU (Rectified Linear Unit) activation function to each element in the input array.

    The ReLU function outputs the input directly if it is positive; otherwise, it outputs zero.

    Parameters:
        x (np.ndarray): A numpy array of any shape containing numeric data.

    Returns:
        np.ndarray: An array of the same shape as `x`, where each element is the result of applying the ReLU function.

    Raises:
        TypeError: If the input `x` is not a numpy array.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    return np.maximum(0, x)


def softmax(logits, axis=1):
    """
    Compute the softmax of a set of scores (logits).

    Args:
    logits (ndarray): Input array containing raw scores for each class.
    axis (int): Axis over which to perform the softmax, typically the feature or class dimension.

    Returns:
    ndarray: Softmax probabilities which sum to 1 along the specified axis.
    """
    # Stabilize the logits by subtracting the maximum value within the axis to prevent overflow in exp
    shift_logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shift_logits)

    # Normalize the exponential scores to get probabilities
    softmax_probs = exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)
    return softmax_probs


def cross_entropy(predictions, labels):
    # Apply softmax to the raw predictions to transform them into probabilities
    predictions = softmax(predictions)

    # Convert labels to one-hot encoded vectors based on the number of classes
    labels = one_hot(labels, predictions.shape[1])

    # Small constant to prevent numerical issues in logarithm calculation
    epsilon = 1e-10

    # Clip predictions to avoid log(0) which results in -inf
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    # Calculate cross-entropy loss
    cross_entropy_loss = -np.sum(labels * np.log(predictions), axis=1)
    return np.mean(cross_entropy_loss)


class MultiLayerNN:
    def __init__(self, input_size, output_size=10, lr=1e-3):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.initialize_weights()

    def initialize_weights(self):
        self.w1 = np.random.randn(self.input_size, 128) * 0.01
        self.b1 = np.zeros(128)
        self.w2 = np.random.randn(128, 64) * 0.01
        self.b2 = np.zeros(64)
        self.w3 = np.random.randn(64, self.output_size) * 0.01
        self.b3 = np.zeros(self.output_size)

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        self.z1 = x @ self.w1 + self.b1
        x = relu(self.z1)
        self.z2 = x @ self.w2 + self.b2
        x = relu(self.z2)
        self.z3 = x @ self.w3 + self.b3
        return softmax(self.z3, axis=1)

    def __call__(self, inputs):
        return self.forward(inputs)

    def parameters(self):
        yield self.w1
        yield self.b1
        yield self.w2
        yield self.b2

    def named_parameters(self):
        params = {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}
        return params.items()

    def _train_step(self, images, labels, optimizer=None):
        batch_size = images.shape[0]
        # forward
        outputs = self(images)

        # calculate loss
        loss = cross_entropy(outputs, labels)

        # backpropagation
        delta_3 = outputs - one_hot(labels, num_classes=10)
        grad_w3 = relu(self.z2).T @ delta_3 / batch_size
        grad_b3 = np.sum(delta_3, axis=0) / batch_size

        delta_2 = delta_3 @ self.w3.T
        delta_2[self.z2 <= 0] = 0
        grad_w2 = relu(self.z1).T @ delta_2 / batch_size
        grad_b2 = np.sum(delta_2, axis=0) / batch_size

        delta_1 = delta_2 @ self.w2.T
        delta_1[self.z1 <= 0] = 0
        grad_w1 = images.reshape(-1, self.input_size).T @ delta_1 / batch_size
        grad_b1 = np.sum(delta_1, axis=0) / batch_size

        # parameters update
        if optimizer is None:
            self.w3 -= self.lr * grad_w3
            self.b3 -= self.lr * grad_b3
            self.w2 -= self.lr * grad_w2
            self.b2 -= self.lr * grad_b2
            self.w1 -= self.lr * grad_w1
            self.b1 -= self.lr * grad_b1
        else:
            optimizer.step([grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3])

        return loss

    def train_and_evaluate(self, epochs, train_dl, test_dl):
        history = []
        optim = Adam(self.parameters())
        for epoch in range(epochs):
            train_losses = []
            for images, labels in train_dl:
                train_losses.append(self._train_step(images, labels, optim))

            test_losses = []
            test_accs = []
            for images, labels in test_dl:
                outputs = self(images)
                test_losses.append(cross_entropy(outputs, labels))
                preds = np.argmax(outputs, axis=1)
                test_accs.append(
                    np.array(np.sum(preds == labels).item() / len(preds)))

            history.append({
                'train_loss': np.stack(train_losses).mean().item(),
                'test_loss': np.stack(test_losses).mean().item(),
                'test_acc': np.stack(test_accs).mean()
            })
            print(
                f"Epoch[{epoch + 1:d}]: train_loss: {history[-1]['train_loss']:.4f}, test_loss: {history[-1]['test_loss']:.4f}, test_acc: {history[-1]['test_acc']:.4f}"
            )
        return history


class Adam():
    def __init__(self, parameters, learning_rate=1e-2, momentum_factors=(0.9, 0.999), stability_constant=1e-8):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.momentum_factors = momentum_factors
        self.stability_constant = stability_constant
        self.time_step = 0
        self.first_moment = [np.zeros_like(param) for param in self.parameters]
        self.second_moment = [np.zeros_like(param) for param in self.parameters]

    def step(self, gradients):
        self.time_step += 1
        for index, param, grad in zip(range(len(gradients)), self.parameters, gradients):
            self.first_moment[index] = self.momentum_factors[0] * self.first_moment[index] + (
                        1 - self.momentum_factors[0]) * grad
            self.second_moment[index] = self.momentum_factors[1] * self.second_moment[index] + (
                        1 - self.momentum_factors[1]) * (grad ** 2)

            corrected_first_moment = self.first_moment[index] / (1 - self.momentum_factors[0] ** self.time_step)
            corrected_second_moment = self.second_moment[index] / (1 - self.momentum_factors[1] ** self.time_step)

            param -= self.learning_rate * corrected_first_moment / (
                        np.sqrt(corrected_second_moment) + self.stability_constant)


batch_size = 64
epochs = 20


train_images = load_mnist_images('./data/mnist/train/train-images.idx3-ubyte')
train_labels = load_mnist_labels('./data/mnist/train/train-labels.idx1-ubyte')
test_images = load_mnist_images('./data/mnist/test/t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('./data/mnist/test/t10k-labels.idx1-ubyte')

train_images = mnist_transform(train_images)
test_images = mnist_transform(test_images)

train_dl = MyDataLoader((train_images, train_labels), batch_size, True)
test_dl = MyDataLoader((test_images, test_labels), batch_size)

model = MultiLayerNN(784)
history = model.train_and_evaluate(epochs, train_dl, test_dl)
plot_history(history)
