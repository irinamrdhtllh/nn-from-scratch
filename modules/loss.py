import numpy as np

from modules.activation import Softmax


class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

    def regularization_loss(self, layer):
        regularization_loss = 0

        # L1 regularization - weights
        if layer.weight_lambda_l1 > 0:
            regularization_loss += layer.weight_lambda_l1 * np.sum(
                np.abs(layer.weights)
            )

        # L1 regularization - biases
        if layer.bias_lambda_l1 > 0:
            regularization_loss += layer.bias_lambda_l1 * np.sum(np.abs(layer.biases))

        # L2 regularization -weights
        if layer.weight_lambda_l2 > 0:
            regularization_loss += layer.weight_lambda_l2 * np.sum(layer.weights**2)

        if layer.bias_lambda_l2 > 0:
            regularization_loss += layer.bias_lambda_l2 * np.sum(layer.biases**2)

        return regularization_loss


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        n_samples = len(y_pred)

        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]

        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples
        n_samples = len(dvalues)
        # Number of labels in every sample
        n_labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]

        # Calculate the gradient
        self.dinputs = -y_true / dvalues
        # Normalize the gradient
        self.dinputs = self.dinputs / n_samples


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
        n_samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete index values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Calculate the gradient
        self.dinputs[range(n_samples), y_true] -= 1
        # Normalize the gradient
        self.dinputs = self.dinputs / n_samples


class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        n_samples = len(dvalues)
        # Number of output in every sample
        n_outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate and normalize the gradient
        self.dinputs = (
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))
            / n_outputs
        )
        self.dinputs = self.dinputs / n_samples


class MeanSquaredError:
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        n_samples = len(dvalues)
        # Number of outputs in every samples
        n_outputs = len(dvalues[0])

        # Calculate and normalize the gradient
        self.dinputs = -2 * (y_true - dvalues) / n_outputs
        self.dinputs = self.dinputs / n_samples
