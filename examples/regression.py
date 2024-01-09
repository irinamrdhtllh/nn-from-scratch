import nnfs
from nnfs.datasets import sine_data

from modules.model import Model
from modules.layer import Dense
from modules.activation import ReLU, Linear
from modules.loss import MeanSquaredError
from modules.optimizer import Adam
from modules.accuracy import Regression

nnfs.init()


# Create the train dataset
X_train, y_train = sine_data()
X_val, y_val = sine_data()

# Instantiate the model
model = Model()

# Add layers
model.add(Dense(1, 64))
model.add(ReLU())
model.add(Dense(64, 64))
model.add(ReLU())
model.add(Dense(64, 1))
model.add(Linear())

# Set loss and optimizer objects
model.set(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Regression(),
)

# Finalize the model
model.finalize()

# Train the model
model.train(
    X_train, y_train, epochs=10000, print_every=100, validation_data=(X_val, y_val)
)
