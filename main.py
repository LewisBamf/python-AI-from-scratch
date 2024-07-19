import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neurone:
    def __init__(self, input_size):
        self.learning_rate = 0.5
        self.weights = np.random.uniform(-1, 1, (input_size,))  # Initialize weights as a 1D array
        self.bias = np.random.uniform(-1, 1)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(np.dot(self.inputs, self.weights) + self.bias)
        return self.output

    def backward(self, error):
        gradient = error * sigmoid_derivative(self.output)
        weight_update = self.learning_rate * gradient * self.inputs
        self.weights += weight_update
        self.bias += self.learning_rate * gradient
        return gradient * self.weights

class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neurone(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, errors):
        return np.sum([neuron.backward(error) for neuron, error in zip(self.neurons, errors)], axis=0)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def train(self, x, y):
        output = self.forward(x)
        error = y - output
        self.backward(error)
        return np.mean(np.abs(error))

# Read the data
path = "house_data.csv"
data = pd.read_csv(path)

# Calculate lengths for training and testing data
training_data_length = int((len(data) * 80) / 100)  # 80% for training
testing_data_length = len(data) - training_data_length

# Split data into training and testing sets
training_data = data.iloc[:training_data_length, :]
testing_data = data.iloc[training_data_length:, :]

# Initialize the MinMaxScaler for features and targets
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Assuming the last column is the target variable and all other columns are features
training_X = training_data.iloc[:, :-1].values
training_Y = training_data.iloc[:, -1].values.reshape(-1, 1)
testing_X = testing_data.iloc[:, :-1].values
testing_Y = testing_data.iloc[:, -1].values.reshape(-1, 1)

# Fit the scaler on the training data and transform both training and testing data
feature_scaler.fit(training_X)
training_X = feature_scaler.transform(training_X)
testing_X = feature_scaler.transform(testing_X)

target_scaler.fit(training_Y)
training_Y = target_scaler.transform(training_Y)
testing_Y = target_scaler.transform(testing_Y)

print(f"training_X shape: {training_X.shape}")
print(f"training_Y shape: {training_Y.shape}")

# Initialize neural network
layer_sizes = [2, 64, 1]
nn = NeuralNetwork(layer_sizes)

# Training the model
epoch = 100

def train_model(epoch):
    for i in range(epoch):
        total_error = 0
        for x, y in zip(training_X, training_Y):
            error = nn.train(x, y.flatten())  # Flatten y to make it 1D
            total_error += error
        print(f'Epoch {i + 1}/{epoch}, Average Absolute Error: {total_error / len(training_X)}')

train_model(epoch)

# Test the model
predictions = np.array([nn.forward(x) for x in testing_X])

# Calculate mean squared error
mse = np.mean((testing_Y - predictions) ** 2)
accuracy = 100 - mse * 100  # Simple accuracy calculation for demonstration

print(f"Mean Squared Error on test data: {mse:.4f}")
print(f"Accuracy on test data: {accuracy:.2f}%")

# Inverse transform the predictions and actual values to get the original prices
original_testing_Y = target_scaler.inverse_transform(testing_Y)
original_testing_X = feature_scaler.inverse_transform(testing_X)
original_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))

# Example prediction
index = 1  # Starting index
examples = 3

for i in range(examples):
    # Access the row directly
    example_input = original_testing_X[index]
    example_true_output = original_testing_Y[index]
    example_prediction = original_predictions[index]

    # Display the example prediction
    print(f"Example prediction:")
    print(f"Area: {example_input[0]}, Bedrooms: {example_input[1]}")
    print(f"Predicted Price: {example_prediction[0]}")
    print(f"Actual Price: {example_true_output[0]}")
    print(f"Out by: {example_true_output[0] - example_prediction[0]}")

    index += 1
