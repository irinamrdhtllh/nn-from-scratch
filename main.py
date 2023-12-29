import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layer import Dense
from activation import ReLU, Softmax
from loss import CategoricalCrossEntropy

nnfs.init()


if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Dense(2, 16)
    activation1 = ReLU()
    dense2 = Dense(16, 3)
    activation2 = Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = CategoricalCrossEntropy()
    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    print("Loss:", loss)
    print("Accuracy:", accuracy)
