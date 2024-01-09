import nnfs
from nnfs.datasets import spiral_data

from modules.model import Model
from modules.layer import Dense, Dropout
from modules.activation import ReLU, Softmax
from modules.loss import CategoricalCrossEntropy
from modules.optimizer import Adam
from modules.accuracy import Categorical

nnfs.init()


X_train, y_train = spiral_data(samples=1000, classes=3)
X_val, y_val = spiral_data(samples=100, classes=3)

model = Model()

model.add(Dense(2, 512, weight_lambda_l2=5e-4, bias_lambda_l2=5e-4))
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Dense(512, 3))
model.add(Softmax())

model.set(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Categorical(binary=False),
)

model.finalize()

model.train(
    X_train, y_train, epochs=10000, print_every=100, validation_data=(X_val, y_val)
)
