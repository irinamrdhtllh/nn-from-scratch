import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layer import Dense
from activation import ReLU, Softmax
from loss import CategoricalCrossEntropy

nnfs.init()


if __name__ == "__main__":
    # Create the dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create the model
    dense1 = Dense(2, 16)
    activation1 = ReLU()
    dense2 = Dense(16, 3)
    activation2 = Softmax()

    # Create the loss function
    loss_function = CategoricalCrossEntropy()

    # Helper variables
    lowest_loss = 1e6
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    for iteration in range(10_000):
        # Update weights and biases with small random values
        dense1.weights += 0.05 * np.random.randn(2, 16)
        dense1.biases += 0.05 * np.random.randn(1, 16)
        dense2.weights += 0.05 * np.random.randn(16, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        # Perform a forward pass of the training data through layers
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        # Calculate the loss
        loss = loss_function.calculate(activation2.output, y)

        # Calculate the accuracy
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        # If loss is smaller, print and save weights and biases aside
        if loss < lowest_loss:
            print(
                "Lower loss is found!",
                "iteration:",
                iteration,
                "loss:",
                loss,
                "accuracy:",
                accuracy,
            )
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
        # Revert weight and biases
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()
