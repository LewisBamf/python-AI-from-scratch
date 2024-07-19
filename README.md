# python-AI-from-scratch

A simple neural network built from scratch in Python to predict house prices based on features like area and the number of bedrooms. The project includes data preprocessing, model training, and evaluation, showcasing fundamental machine learning concepts without relying on high-level libraries.

## Project Overview

This project demonstrates the implementation of a basic neural network from scratch using only core Python libraries like NumPy and Pandas. The neural network is designed to predict house prices based on features such as the size of the house (in square feet) and the number of bedrooms.

### Key Components

1. **Data Preprocessing**:
   - Reading the house data from a CSV file.
   - Splitting the data into training and testing sets.
   - Normalizing the features and target values using MinMaxScaler.

2. **Neural Network Architecture**:
   - **Neurone Class**: Defines a single neuron, including methods for forward and backward propagation.
   - **Layer Class**: Manages a layer of neurons, handling the propagation of inputs through the neurons in the layer.
   - **NeuralNetwork Class**: Constructs the neural network by stacking layers, facilitating the forward and backward propagation through the entire network.

3. **Training and Evaluation**:
   - Training the neural network using the training dataset.
   - Evaluating the performance on the testing dataset.
   - Providing example predictions and comparing them to actual house prices.

### Challenge: Multiplying Different-Sized Matrices

One of the significant challenges in this project was handling the multiplication of different-sized matrices, which is crucial in neural network computations.

#### The Problem

In a neural network, weights and inputs often have different dimensions. For instance, in a fully connected layer, the input might be a vector of size `n`, and the weights matrix might be of size `m x n`, where `m` is the number of neurons in the layer. Correctly aligning and multiplying these matrices is essential for accurate forward and backward propagation.

#### The Solution

To address this challenge, I used NumPy's dot product functionality, which simplifies matrix multiplication and ensures correct dimensional alignment. Here’s how it was implemented:

- **Forward Propagation**:
  - Each neuron's output is calculated as the dot product of the input vector and the weights, added to the bias.
  ```python
  def forward(self, inputs):
      self.inputs = inputs
      self.output = sigmoid(np.dot(self.inputs, self.weights) + self.bias)
      return self.output
**Backward Propagation**:
- The gradients are calculated and used to update the weights and bias.

```python
def backward(self, error):
    gradient = error * sigmoid_derivative(self.output)
    weight_update = self.learning_rate * gradient * self.inputs
    self.weights += weight_update
    self.bias += self.learning_rate * gradient
    return gradient * self.weights
```

By carefully managing the shapes and dimensions of matrices during these operations, the neural network can efficiently learn from the training data and make accurate predictions.

### Getting Started
**Prerequisites**
 - Python 3.x
 - NumPy
 - Pandas
 - Scikit-learn
 - Running the Project
 - Clone the Repository:

  ```bash
git clone https://github.com/LewisBamf/python-AI-from-scratch.git
cd python-AI-from-scratch
  ```
Install the Required Packages:

  ```bash
pip install numpy pandas scikit-learn
  ```
Run the Script:

  ```bash
python script_name.py
Replace script_name.py with the name of your Python script.
  ```
