import numpy as np


class Accuracy:
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate the accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        # Calculate the accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy


class Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def initialize(self, y, reinit=False):
        # Calculate precision value based on the passed in ground truth
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Categorical(Accuracy):
    def __init__(self, binary=False):
        self.binary = binary

    def initialize(self, y):
        # No initialization is needed
        pass

    def compare(self, predictions, y):
        if not self.binary:
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
        return predictions == y
