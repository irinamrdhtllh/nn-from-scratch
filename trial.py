import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from modules.activation import ReLU
from modules.layer import Dense
from modules.loss import Softmax_CategoricalCrossEntropy
from modules.optimizer import SGD

nnfs.init()


def trial():
    X, y = spiral_data(samples=100, classes=3)

    # Create the 1st Dense layer with 2 input features (x, y) and 128 output values
    dense1 = Dense(2, 64)
    # Create ReLU activation for the 1st layer
    activation1 = ReLU()
    # Create the 2nd Dense layer with 128 input features and 3 output values (same as the number of the data classes)
    dense2 = Dense(64, 3)

    # Create Softmax classifier's combined activation and loss
    activation_loss = Softmax_CategoricalCrossEntropy()

    # Create optimizer
    optimizer = SGD(decay=1e-3)

    for epoch in range(10_000 + 1):
        # Perform a forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        loss = activation_loss.forward(dense2.output, y)

        # Calculate accuracy from activation2's output and target
        predictions = np.argmax(activation_loss.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(
                f"epoch: {epoch}, "
                + f"acc: {accuracy:.3f}, "
                + f"loss: {loss:.3f}, "
                + f"lr: {optimizer.current_learning_rate}"
            )

        # Backward pass
        activation_loss.backward(activation_loss.output, y)
        dense2.backward(activation_loss.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update wweights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()


if __name__ == "__main__":
    trial()
