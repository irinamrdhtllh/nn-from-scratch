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

    def set(self, *, loss, optimizer, accuracy):
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

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.initialize(y)

        # Main training loop
        for epoch in range(1, epochs + 1):
            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss_func.calculate(
                output, y, include_regularization=True
            )
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform the backward pass
            self.backward(output, y)

            # Update parameters
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(
                    f"epoch: {epoch}, "
                    + f"acc: {accuracy:.3f}, "
                    + f"loss: {loss:.3f}, "
                    + f"data_loss: {data_loss:.3f}, "
                    + f"reg_loss: {regularization_loss:.3f}, "
                    + f"lr: {self.optimizer.current_learning_rate}"
                )

        if validation_data is not None:
            X_val, y_val = validation_data

            # Perform the forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss_func.calculate(output, y_val)

            # Get predictions and calculate the accuracy
            predictions = self.output_layer.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")
