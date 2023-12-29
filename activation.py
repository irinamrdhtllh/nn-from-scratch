import numpy as np


class ReLU:
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


class Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.output = probabilities
