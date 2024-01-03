import numpy as np


class Dense:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_lambda_l1=0,
        weight_lambda_l2=0,
        bias_lambda_l1=0,
        bias_lambda_l2=0,
    ):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_lambda_l1 = weight_lambda_l1
        self.weight_lambda_l2 = weight_lambda_l2
        self.bias_lambda_l1 = bias_lambda_l1
        self.bias_lambda_l2 = bias_lambda_l2

    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_lambda_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_lambda_l1 * dL1
        # L1 on biases
        if self.bias_lambda_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_lambda_l1 * dL1
        # L2 on weights
        if self.weight_lambda_l2 > 0:
            self.dweights += 2 * self.weight_lambda_l2 * self.weights
        # L2 on biases
        if self.bias_lambda_l2 > 0:
            self.dbiases += 2 * self.bias_lambda_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
