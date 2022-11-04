import numpy as np
import random
import json
import sys

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return (1.0 / (1.0 + np.exp(-x)) - 1.0) * (1.0 / (1.0 + np.exp(-x)))

class Network:
    #sizes is the list of number of nodes in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes) 
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] 

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
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
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                # feedback after every epoch TODO: maybe too inefficiant 
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print(f"Epoch {j} complete")

    # calculate error
    def cost_derivative(self, output_activations, expected_output):
        return(output_activations - expected_output)

    def backprop(self, input, expected_output):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = input
        activations = [input] # stores activations layer by layer
        zs = [] # stores z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
       
        # backward pass
        delta = self.cost_derivative(activations[-1], expected_output) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # backpropagate through every layer
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            nabla_b[-_layer] = delta
            nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # sum them all up
        for input, expected_output in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(input, expected_output)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #update the weights and biases
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        #testing the ANN
        test_results = [(np.argmax(self.feed_forward(input)), expected_output) for (input, expected_output) in test_data]
        return sum(int(y==x) for (x, y) in test_results)

    def export(self, filename):
        data = {"sizes": self.sizes,
        "weights": [w.tolist() for w in self.weights],
        "biases": [b.tolist() for b in self.biases]} 
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
    
    def load(self, filename):
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        self.sizes = data["sizes"]
        self.weights = data["weights"]
        self.biases = data["biases"]

