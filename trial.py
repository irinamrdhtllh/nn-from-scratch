import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from modules.activation import ReLU
from modules.layer import Dense
from modules.loss import Softmax_CategoricalCrossEntropy
from modules.optimizer import SGD, AdaGrad, RMSProp, Adam

nnfs.init()


def trial():
    X, y = spiral_data(samples=1_000, classes=3)

    # Create the 1st Dense layer with 2 input features (x, y) and 256 output values
    dense1 = Dense(2, 512, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
    # Create ReLU activation for the 1st layer
    activation1 = ReLU()
    # Create the 2nd Dense layer with 256 input features and 3 output values (same as the number of the data classes)
    dense2 = Dense(512, 3)

    # Create Softmax classifier's combined activation and loss
    activation_loss = Softmax_CategoricalCrossEntropy()

    # Create optimizer
    optimizer = Adam(learning_rate=0.05, decay=5e-7)

    # Train the model
    for epoch in range(10_000 + 1):
        # Perform a forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        data_loss = activation_loss.forward(dense2.output, y)

        # fmt: off
        regularization_loss = (
            activation_loss.loss.regularization_loss(dense1) 
            + activation_loss.loss.regularization_loss(dense2)
        )
        # fmt: on

        loss = data_loss + regularization_loss

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
                + f"data_loss: {data_loss: .3f}, "
                + f"reg_loss: {regularization_loss: .3f}, "
                + f"lr: {optimizer.current_learning_rate}",
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

    # Validate the model
    X_test, y_test = spiral_data(samples=100, classes=3)

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
    trial()
