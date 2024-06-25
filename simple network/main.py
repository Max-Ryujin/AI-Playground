import mnist_loader
import network  # Assuming the original Network class is here
import secret_network  # Assuming the modified Network class is here
import numpy as np

def main():
    # Load the MNIST data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Define hyperparameters to compare
    hyperparameters = [
        {"sizes": [784, 30, 10], "epochs": 10, "mini_batch_size": 10, "eta": 3.0},
        {"sizes": [784, 30, 10], "epochs": 20, "mini_batch_size": 10, "eta": 3.0},
        {"sizes": [784, 30, 10], "epochs": 10, "mini_batch_size": 20, "eta": 3.0},
        {"sizes": [784, 30, 10], "epochs": 10, "mini_batch_size": 10, "eta": 1.0},
    ]

    # Compare performance
    for params in hyperparameters:
        print(f"\nHyperparameters: {params}")
        
        # Initialize the original network
        original_net = network.Network(params["sizes"])
        print("Training original network...")
        original_net.SGD(training_data, params["epochs"], params["mini_batch_size"], params["eta"], test_data=test_data)
        
        # Initialize the modified network
        modified_net = secret_network.Network(params["sizes"])
        print("Training modified network...")
        modified_net.SGD(training_data, params["epochs"], params["mini_batch_size"], params["eta"], test_data=test_data)

if __name__ == "__main__":
    main()
