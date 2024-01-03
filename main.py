import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from modules.activation import ReLU
from modules.layer import Dense
from modules.loss import Softmax_CategoricalCrossEntropy
from modules.optimizer import Adam

nnfs.init()


def train(X_train, y_train):
    # Train the model
    for epoch in range(10000 + 1):
        # Perform a forward pass
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        data_loss = activation_loss.forward(dense2.output, y_train)

        # fmt: off
        regularization_loss = (
            activation_loss.loss.regularization_loss(dense1) 
            + activation_loss.loss.regularization_loss(dense2)
        )
        # fmt: on

        loss = data_loss + regularization_loss

        # Calculate accuracy from activation2's output and target
        predictions = np.argmax(activation_loss.output, axis=1)
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == y_train)

        if not epoch % 100:
            print(
                f"epoch: {epoch}, "
                + f"acc: {accuracy:.3f}, "
                + f"loss: {loss:.3f}, "
                + f"data_loss: {data_loss: .3f}, "
                + f"reg_loss: {regularization_loss: .3f}, "
                + f"lr: {optimizer.current_learning_rate}",
            )

        # Backward pass
        activation_loss.backward(activation_loss.output, y_train)
        dense2.backward(activation_loss.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update wweights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()


def test(X_test, y_test):
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss = activation_loss.forward(dense2.output, y_test)

    predictions = np.argmax(activation_loss.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)

    print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")


if __name__ == "__main__":
    # Create the dataset
    X_train, y_train = spiral_data(samples=1000, classes=3)
    X_test, y_test = spiral_data(samples=100, classes=3)

    # Create the model
    dense1 = Dense(2, 512, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
    activation1 = ReLU()
    dense2 = Dense(512, 3)
    activation_loss = Softmax_CategoricalCrossEntropy()
    optimizer = Adam(learning_rate=0.02, decay=5e-7)

    # Train the model
    train(X_train, y_train)

    # Validate the model
    test(X_test, y_test)
