import os
import cv2
import nnfs
import numpy as np

from modules.model import Model
from modules.layer import Dense
from modules.activation import ReLU, Softmax
from modules.loss import CategoricalCrossEntropy
from modules.optimizer import Adam
from modules.accuracy import Categorical

nnfs.init()


def load_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED,
            )
            X.append(image)
            y.append(label)

    # Convert the data to numpy array
    return np.array(X), np.array(y).astype("uint8")


def create_data(path):
    X_train, y_train = load_dataset("train", path)
    X_test, y_test = load_dataset("test", path)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    path = "examples/fashion_mnist"
    X_train, y_train, X_test, y_test = create_data(os.path.join(path, "images"))

    # Shuffle the training dataset
    keys = np.array(range(X_train.shape[0]))
    np.random.shuffle(keys)
    X_train = X_train[keys]
    y_train = y_train[keys]

    # Reshape to vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Scale images to be between the range of -1 and 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    # Create the model
    model = Model()

    model.add(Dense(X_train.shape[1], 128))
    model.add(ReLU())
    model.add(Dense(128, 128))
    model.add(ReLU())
    model.add(Dense(128, 10))
    model.add(Softmax())

    # Set loss, optimizer, and accuracy
    model.set(
        loss=CategoricalCrossEntropy(),
        optimizer=Adam(decay=1e-3),
        accuracy=Categorical(),
    )

    model.finalize()

    # Train the model
    model.train(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
        print_every=100,
        validation_data=(X_test, y_test),
    )

    # Save parameters
    model.save_parameters(os.path.join(path, "model_params.parms"))

    # Load the parameters
    model.load_parameters(os.path.join(path, "model_params.parms"))

    # Evaluate the model
    model.evaluate(X_test, y_test)
