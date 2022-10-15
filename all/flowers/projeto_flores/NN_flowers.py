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
data_X = pd.DataFrame(np.array(data).T[:4].T, dtype='float')
data_Y = pd.DataFrame(data).T[4:].T
print(data_Y)


class NN():

    def __init__(self, shape):
        batches, n_features = np.shape(shape)
        n_neurons = 4
        np.random.seed(0)

        self.weights = np.random.randn(n_features, n_neurons)
        self.bias = np.zeros(n_neurons)
        # cria 4 neurônios com "n_features" de valores para multiplicar cada entrada

        self.weights_2 = 0.10*np.random.randn(n_neurons, n_neurons)
        self.bias_2 = np.zeros(n_neurons)
        # cria 4 neurônios com "n_features" de valores para multiplicar cada entrada

    def feedForward(self, X_train):
        output = np.dot(X_train, self.weights) + self.bias
        output_2 = np.dot(output, self.weights_2) + self.bias_2
        return output_2


# entra com a batch pra definir os parâmetros da rede
flowers = NN(data_X)

result = flowers.feedForward(data_X)
print(pd.DataFrame(result))

# weights = pd.DataFrame(np.random.randn(10, 4).T)
# print(data_X)
# print(weights)
# result = np.dot(data_X, weights).T


# for i in range(10):
#     print("\n")

#     # faz o processamento da entrada de data_X e produz uma saída de mesma dimensão da batch
#     result = flowers.feedForward(data_X)
#     print(pd.DataFrame(result))
