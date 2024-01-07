import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data

from modules.layer import Dense
from modules.activation import ReLU, Linear
from modules.loss import MeanSquaredError
from modules.optimizer import Adam

nnfs.init()


# Initialize the train dataset
X_train, y_train = sine_data()

# Create the model
dense1 = Dense(1, 64)
activation1 = ReLU()
dense2 = Dense(64, 64)
activation2 = ReLU()
dense3 = Dense(64, 1)
activation3 = Linear()
loss_func = MeanSquaredError()
optimizer = Adam(learning_rate=0.02, decay=1e-3)

# Calculate the accuracy precision
precision = np.std(y_train) / 250

# ==============================
#        Train the model
# ==============================

for epoch in range(10000 + 1):
    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Calculate the data loss
    data_loss = loss_func.calculate(activation3.output, y_train)

    # Calculate the regularization penalty
    # fmt: off
    regularization_loss = (
        loss_func.regularization_loss(dense1)
        + loss_func.regularization_loss(dense2)
        + loss_func.regularization_loss(dense3)
    )
    # fmt: on

    # Calculate the overall loss
    loss = data_loss + regularization_loss

    # Calculate model's predictions and accuracy
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y_train) < precision)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, "
            + f"acc: {accuracy:.3f}, "
            + f"loss: {loss:.3f}, "
            + f"data_loss: {data_loss:.3f}, "
            + f"reg_loss: {regularization_loss:.3f}, "
            + f"lr: {optimizer.current_learning_rate}"
        )

    # Backward pass
    loss_func.backward(activation3.output, y_train)
    activation3.backward(loss_func.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update model's weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
