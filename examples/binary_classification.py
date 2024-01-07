import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from modules.layer import Dense
from modules.activation import ReLU, Sigmoid
from modules.loss import BinaryCrossEntropy
from modules.optimizer import Adam

nnfs.init()


# Initialize the train dataset
X_train, y_train = spiral_data(samples=100, classes=2)
y_train = y_train.reshape(-1, 1)

# Create the model
dense1 = Dense(2, 64, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4)
activation1 = ReLU()
dense2 = Dense(64, 1)
activation2 = Sigmoid()
loss_func = BinaryCrossEntropy()
optimizer = Adam(decay=5e-7)

# ==============================
#        Train the model
# ==============================

for epoch in range(10000 + 1):
    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate the data loss
    data_loss = loss_func.calculate(activation2.output, y_train)

    # Calculate the regularization loss
    # fmt: off
    regularization_loss = (
        loss_func.regularization_loss(dense1)
        + loss_func.regularization_loss(dense2)
    )
    # fmt: on

    # Calculate the overall loss
    loss = data_loss + regularization_loss

    # Calculate the predictions and accuracy
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y_train)

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
    loss_func.backward(activation2.output, y_train)
    activation2.backward(loss_func.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update model's weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# ==============================
#       Validate the model
# ==============================

# Create the test dataset
X_test, y_test = spiral_data(samples=100, classes=2)
y_test = y_test.reshape(-1, 1)

# Forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Calculate the data loss
loss = loss_func.calculate(activation2.output, y_test)

# Calculate the predictions and accuracy
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f"test, acc: {accuracy:.3f}, loss: {loss:.3f}")
