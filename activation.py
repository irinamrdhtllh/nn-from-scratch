import numpy as np

from loss import CategoricalCrossEntropy


class ReLU:
    def forward(self, inputs):
        # Calculate output values from inputs
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient if the input value is negative
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for i, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            output = output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            # Calculate sample-wise gradient
            self.dinputs[i] = np.dot(jacobian_matrix, dvalue)


class Softmax_CategoricalCrossEntropy:
    """Combined softmax activation and categorical cross entropy loss for faster backward step"""

    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        loss = self.loss.calculate(self.output, y_true)

        return loss

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete index values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Calculate the gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize the gradient
        self.dinputs = self.dinputs / samples
