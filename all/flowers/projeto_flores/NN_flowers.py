import numpy as np
import pandas as pd

data = pd.read_csv('./data.txt')
data_test = [[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
             [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
             [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
             [4.6, 3.1, 1.5, 0.2, 'Iris-setosa'],
             [5.0, 3.6, 1.4, 0.2, 'Iris-setosa'],
             [5.4, 3.9, 1.7, 0.4, 'Iris-setosa'],
             [4.6, 3.4, 1.4, 0.3, 'Iris-setosa'],
             [5.0, 3.4, 1.5, 0.2, 'Iris-setosa'],
             [4.4, 2.9, 1.4, 0.2, 'Iris-setosa'],
             [4.9, 3.1, 1.5, 0.1, 'Iris-setosa']]

# batch de 10 elementos para teste
data_X = np.array(data).T[:4].T
data_Y = np.array(pd.DataFrame(data).T[4:].T)
data_test = np.array(pd.DataFrame(data_test).T[:4].T)
data_X = data_X.astype(float)
data_test = data_test.astype(float)


class NN():

    def __init__(self, shape):
        batches, n_features = np.shape(shape)
        n_neurons = 4
        np.random.seed(0)

        self.weights = np.random.randn(n_features, n_neurons)
        self.bias = np.zeros(n_neurons)
        # cria 4 neurônios com "n_features" de valores para multiplicar cada entrada dos dados iniciais

        self.weights_2 = 0.10*np.random.randn(n_neurons, n_neurons)
        self.bias_2 = np.zeros(n_neurons)
        # cria 4 neurônios com "n_features" de valores para multiplicar cada entrada da primeira camada

        self.weights_3 = 0.10*np.random.randn(n_neurons, 3)
        self.bias_3 = np.zeros(3)
        # cria 3 neurônios com "n_features" de valores para multiplicar cada entrada da segunda camada

    def ReLU(self, input):
        return np.maximum(0, input)

    def softmax(self, input):
        output = np.exp(input - np.max(input, axis=1, keepdims=True))
        normalized = output/np.sum(output, axis=1, keepdims=True)
        return normalized

    def feedForward(self, X_train):
        output = self.ReLU(np.dot(X_train, self.weights) + self.bias)
        output_2 = self.ReLU(np.dot(output, self.weights_2) + self.bias_2)
        result = self.softmax(np.dot(output_2, self.weights_3) + self.bias_3)
        return result


# entra com a batch pra definir os parâmetros da rede
flowers = NN(data_X)

result = flowers.feedForward(data_X)
print(pd.DataFrame(result))

