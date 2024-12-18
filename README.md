# nn-from-scratch

This repository contains an implementation of a simple neural network built from scratch using only Python and NumPy. 

The project is organized into two main folders:

## Modules
The `modules` folder contains key building blocks of the neural network:
- `accuracy.py`: Implements classes for calculating accuracy for both regression and classification problems.
- `activation.py`: Defines various activation layers, including ReLU, softmax, sigmoid, and linear.
- `layer.py`: Contains classes for different layers, such as input, dense, and dropout layers.
- `loss.py`: Implements loss functions such as categorical cross-entropy, binary cross-entropy, mean squared error (MSE), and mean absolute error (MAE)
- `model.py`: Contains the model class with methods for forward pass, backward pass, training, and evaluation
- `optimizer.py`: Implements optimizers including stochastic gradient descent (SGD), AdaGrad, RMSProp, and Adam.

## Examples
The examples folder includes implementations of the neural network on various tasks:
- Classification: Using the spiral dataset from `nnfs.datasets`
- Regression: Using the sine wave dataset from `nnfs.datasets`
- Fashion-MNIST: Using the Fashion-MNIST dataset

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/irinamrdhtllh/nn-from-scratch.git
cd nn-from-scratch

```
2. Install requirements from `requirements.txt`:
```bash
pip install -r requirements.txt
```
3. Run specific algorithms or data structures. For example:
```bash
python examples/classification.py
```