
import json
import sys
import random

import numpy as np

class Network(object):
    """A class representing a neural network.

    Attributes:
        num_layers (int): The number of layers in the network.
        sizes (list): A list of integers representing the number of neurons in each layer.
        biases (list): A list of numpy arrays representing the biases of each layer.
        weights (list): A list of numpy arrays representing the weights of each layer.

    Methods:
        __init__(self, sizes): Initializes the network with random biases and weights.
        feedforward(self, a): Performs feedforward propagation to compute the output of the network.
        SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None): Performs stochastic gradient descent to train the network.
        update_mini_batch(self, mini_batch, eta): Updates the network's weights and biases using backpropagation on a mini batch.
        backprop(self, x, y): Performs backpropagation to compute the gradient for the cost function.
        evaluate(self, test_data): Evaluates the network's performance on test data.
        cost_derivative(self, output_activations, y): Computes the derivative of the cost function with respect to the output activations.
        export(self, filename): Exports the network's parameters to a JSON file.
        load(self, filename): Loads the network's parameters from a JSON file.
    """

    def __init__(self, sizes):
        """Initializes the network with random biases and weights.

        Args:
            sizes (list): A list of integers representing the number of neurons in each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Performs feedforward propagation to compute the output of the network,
        adjusting the activation thresholds dynamically.

        Args:
            a (numpy array): The input to the network.

        Returns:
            numpy array: The output of the network.
        """
        for i in range(self.num_layers - 1):
            b, w = self.biases[i], self.weights[i]
            z = np.dot(w, a) + b
            a = sigmoid(z)

            if i < self.num_layers - 2:
                next_w = self.weights[i + 1]
                next_b = self.biases[i + 1]
                next_z = np.dot(next_w, a) + next_b
                next_a = sigmoid(next_z)
                # Compute adjusted biases for the current feedforward pass
                adjusted_b = self.adjust_thresholds(next_a, next_w, b)
                z = np.dot(w, a) + adjusted_b
                a = sigmoid(z)
        return a



    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """Performs stochastic gradient descent to train the network.

        Args:
            train_data (list): A list of tuples (x, y) representing the training inputs and corresponding outputs.
            epochs (int): The number of epochs to train for.
            mini_batch_size (int): The size of each mini batch.
            eta (float): The learning rate.
            test_data (list, optional): A list of tuples (x, y) representing the test inputs and corresponding outputs.

        Returns:
            None
        """
        # initialize data
        training_data = list(train_data)
        samples = len(training_data)

        # for testing
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
       
        # using mini batches 
        for j in range(epochs):
            random.shuffle(training_data) 
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))


    def adjust_thresholds(self, activations_next, weights_next, biases):
        """Adjusts the thresholds (biases) of the current layer based on the next layer's activations.

        Args:
            activations_next (numpy array): The activations of the next layer.
            weights_next (numpy array): The weights connecting the current layer to the next layer.
            biases (numpy array): The biases of the current layer.

        Returns:
            numpy array: The adjusted biases.
        """
        adjustment_factor = 0.1  # scaling factor for adjustment
        adjustment = adjustment_factor * np.dot(weights_next.T, activations_next)
        return biases + adjustment


    def update_mini_batch(self, mini_batch, eta):
        """Updates the network's weights and biases using backpropagation on a mini batch.

        Args:
            mini_batch (list): A list of tuples (x, y) representing the mini batch inputs and corresponding outputs.
            eta (float): The learning rate.

        Returns:
            None
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Performs backpropagation to compute the gradient for the cost function.

        Args:
            x (numpy array): The input to the network.
            y (numpy array): The desired output of the network.

        Returns:
            tuple: A tuple (nabla_b, nabla_w) representing the gradient for the cost function.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Evaluates the network's performance on test data.

        Args:
            test_data (list): A list of tuples (x, y) representing the test inputs and corresponding outputs.

        Returns:
            int: The number of test inputs for which the network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Computes the derivative of the cost function with respect to the output activations.

        Args:
            output_activations (numpy array): The output activations of the network.
            y (numpy array): The desired output of the network.

        Returns:
            numpy array: The vector of partial derivatives of the cost function.
        """
        return (output_activations-y)

    def export(self, filename):
        """Exports the network's parameters to a JSON file.

        Args:
            filename (str): The name of the file to save the parameters to.

        Returns:
            None
        """
        data = {"sizes": self.sizes,
        "weights": [w.tolist() for w in self.weights],
        "biases": [b.tolist() for b in self.biases]} 
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
    
    def load(self, filename):
        """Loads the network's parameters from a JSON file.

        Args:
            filename (str): The name of the file to load the parameters from.

        Returns:
            None
        """
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        self.sizes = data["sizes"]
        self.weights = data["weights"]
        self.biases = data["biases"]

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function.

    Args:
        z (numpy array): The input to the sigmoid function.

    Returns:
        numpy array: The output of the sigmoid function.
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function.

    Args:
        z (numpy array): The input to the sigmoid function.

    Returns:
        numpy array: The derivative of the sigmoid function.
    """
    return sigmoid(z)*(1-sigmoid(z))