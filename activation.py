import numpy as np


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
