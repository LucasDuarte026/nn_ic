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

# batch de 10 elementos para teste
data_ent = [[5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5.0, 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1]]

data = np.array(data)
data_ent = np.array(data_ent)


class NN():

    def __init__(self, shape):
        batches, n_features = np.shape(shape)
        n_neurons = 4

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


flowers = NN(data_ent) # entra com a batch pra definir os parâmetros da rede

for i in range(10):
    print("\n")
    result = flowers.feedForward(data_ent) #faz o processamento da entrada de data_ent e produz uma saída de mesma dimensão da batch
    print(pd.DataFrame(result))