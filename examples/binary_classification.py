import nnfs
from nnfs.datasets import spiral_data

from modules.model import Model
from modules.layer import Dense
from modules.activation import ReLU, Sigmoid
from modules.loss import BinaryCrossEntropy
from modules.optimizer import Adam
from modules.accuracy import Categorical

nnfs.init()


X_train, y_train = spiral_data(samples=100, classes=2)
y_train = y_train.reshape(-1, 1)

X_val, y_val = spiral_data(samples=100, classes=2)
y_val = y_val.reshape(-1, 1)

model = Model()

model.add(Dense(2, 64, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4))
model.add(ReLU())
model.add(Dense(64, 1))
model.add(Sigmoid())

model.set(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(decay=5e-7),
    accuracy=Categorical(binary=True),
)

model.finalize()

model.train(
    X_train, y_train, epochs=10000, print_every=100, validation_data=(X_val, y_val)
)
