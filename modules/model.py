import os
import pickle
import copy
import numpy as np

from modules.layer import Input
from modules.activation import Softmax
from modules.loss import CategoricalCrossEntropy, Softmax_CategoricalCrossEntropy


class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        self.softmax_loss_output = None

    def add(self, layer):
        # Add objects to the model
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        self.loss_func = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        # Create and set the input layer
        self.input_layer = Input()

        # Count all the objects
        n_layers = len(self.layers)

        # Initialize a list containing trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(n_layers):
            # If it is the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # All layers (except the first and the last)
            elif i < (n_layers - 1):
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # The last layer
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss_func
                self.output_layer = self.layers[i]

            # Check if layer is trainable
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss_func is not None:
            self.loss_func.remember_trainable_layers(self.trainable_layers)

        # fmt: off
        if (
            isinstance(self.layers[-1], Softmax) and 
            isinstance(self.loss_func, CategoricalCrossEntropy)
        ):
            self.softmax_loss_output = Softmax_CategoricalCrossEntropy()
        # fmt: on

    def forward(self, X, training):
        # Perform forward pass
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_loss_output is not None:
            # Perform backward pass
            self.softmax_loss_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_loss_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # Perform backward pass
        self.loss_func.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(
        self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None
    ):
        # Initialize accuracy object
        self.accuracy.initialize(y)

        # Default value if batch size is not being set
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):
            print("=" * os.get_terminal_size()[0])
            print(f"epoch: {epoch}")

            # Reset accumulated values in loss and accuracy
            self.loss_func.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss_func.calculate(
                    output, batch_y, include_regularization=True
                )
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform the backward pass
                self.backward(output, batch_y)

                # Update parameters
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(
                        f"step: {step}, "
                        + f"acc: {accuracy:.3f}, "
                        + f"loss: {loss:.3f}, "
                        + f"data_loss: {data_loss:.3f}, "
                        + f"reg_loss: {regularization_loss:.3f}, "
                        + f"lr: {self.optimizer.current_learning_rate}"
                    )

            # Get epoch loss and accuracy
            (
                epoch_data_loss,
                epoch_regularization_loss,
            ) = self.loss_func.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(
                f"training, "
                + f"acc: {epoch_accuracy:.3f}, "
                + f"loss: {epoch_loss:.3f}, "
                + f"data_loss: {epoch_data_loss:.3f}, "
                + f"reg_loss: {epoch_regularization_loss:.3f}, "
                + f"lr: {self.optimizer.current_learning_rate}"
            )

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        val_steps = 1

        if batch_size is not None:
            val_steps = len(X_val) // batch_size
            if val_steps * batch_size < len(X_val):
                val_steps += 1

        # Reset accumulated values in loss and accuracy
        self.loss_func.new_pass()
        self.accuracy.new_pass()

        for step in range(val_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]

            # Perform forward pass
            output = self.forward(batch_X, training=False)

            # Calculate loss
            self.loss_func.calculate(output, batch_y)

            # Get predictions and calculate the accuracy
            predictions = self.output_layer.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get validation loss and accuracy
        val_loss = self.loss_func.calculate_accumulated()
        val_accuracy = self.accuracy.calculate_accumulated()

        print(f"validation, acc: {val_accuracy:.3f}, loss: {val_loss:.3f}")

    def get_parameters(self):
        parameters = []

        # Get parameters from all trainable layers
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters

    def set_parameters(self, parameters):
        # Update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        # Open a file in the binary-write mode and save model's parameters
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        # Open file in the binary-read mode, load weights and update trainable layers
        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in losss and accuracy objects
        model.loss_func.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer and gradient from the loss object
        model.input_layer.__dict__.pop("output", None)
        model.loss_func.__dict__.pop("dinputs", None)

        # Remove all layers' properties
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)

        # Open a file in the binary write-mode and save the model
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        # Open file in the binary-read mode, load the model
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model

    def predict(self, X, *, batch_size=None):
        # Calculate number of steps
        predict_steps = 1
        if batch_size is not None:
            predict_steps = len(X) // batch_size
            if predict_steps * batch_size < len(X):
                predict_steps += 1

        output = []

        for step in range(predict_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size : (step + 1) * batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)
