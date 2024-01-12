import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from modules.model import Model


# Folder path
PATH = "examples/fashion_mnist"

# Label index to label name relation
FASHION_MNIST_LABELS = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


if __name__ == "__main__":
    # Read the image data
    image_data = cv2.imread(
        os.path.join(PATH, "images/inference/t-shirt.png"), cv2.IMREAD_GRAYSCALE
    )

    # Resize the image to the same size as fashion-mnist images
    image_data = cv2.resize(image_data, (28, 28))

    # Invert image colors
    image_data = 255 - image_data

    # Reshape and scale pixel data
    image_data = image_data.reshape(1, -1)
    image_data = (image_data.astype(np.float32) - 127.5) / 127.5

    # Load the trained model
    model = Model.load(os.path.join(PATH, "fashion_mnist.model"))

    # Predict on the image
    confidences = model.predict(image_data)
    predictions = model.output_layer.predictions(confidences)
    prediction = FASHION_MNIST_LABELS[predictions[0]]

    print(prediction)
