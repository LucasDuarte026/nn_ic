from email import feedparser
import pandas as pd
import numpy as np

np.random.seed(0)

data = [[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
        [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
        [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
        [4.6, 3.1, 1.5, 0.2, 'Iris-setosa'],
        [5.0, 3.6, 1.4, 0.2, 'Iris-setosa'],
        [5.4, 3.9, 1.7, 0.4, 'Iris-setosa'],
        [4.6, 3.4, 1.4, 0.3, 'Iris-setosa'],
        [5.0, 3.4, 1.5, 0.2, 'Iris-setosa'],
        [4.4, 2.9, 1.4, 0.2, 'Iris-setosa'],
        [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']]

data = np.array(data)
print(data)


class NN():

    def __init__(self, shape):
        n_features, n_size = shape
        n_neurons=4
        self.weights = np.random.randn(n_features, n_neurons) # cria 4 neurônios com "n_features" de valores para multiplicar cada entrada
        self.weights_2 = np.random.randn(n_neurons, n_neurons)# cria 4 neurônios com "n_features" de valores para multiplicar cada entrada
        # self.bias = np.zeros(n_features, n_size)

    def feedForward(self, X_train):
        output = np.dot(self.weights, X_train)
        output_2 = np.dot(self.weights, output)
        return output_2


# print(data[1].T)
# layer_1 = NN(np.shape(data))

# result = layer_1.feedForward(data[1].T)
# print(result)
