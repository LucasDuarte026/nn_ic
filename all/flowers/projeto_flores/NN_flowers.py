import pandas as pd
import numpy as np

# insere as entradas
# self.input = _data
# inicializa os neurônios de dimenção 10x1 indo de  -1 a 1 em floats randomizados
# self.weights = 2 * np.random.random((10, 1))-1
# self.y = 2 * np.random.random((10, 1))-1

data = pd.read_csv('data.txt').T.to_numpy()

entrada = data[:4].T
for line in range(data[4])
    np.array[y] = data[4]

print(pd.DataFrame(entrada))
# print(data[:4])


class Flowers_NeuralNetwork():  # rede neural de apenas um layer
    def __init__(self, learning_rate=0.01, n_inters=1000):
        np.random.seed(1)
        self.lr = learning_rate
        self.n_inter = n_inters
        self.weights = None
        self.bias = None

    def _activation_func(self, x):
        return np.where(1 if x >= 0 else 0)

# função de treinamento
    # x é a entrada de dados e y é o resultado esperado, ambos em mesma dimensão
    def feedfoward(self, x, y):
        elems, n_features = np.shape(x)

        self.weights = np.zeros(n_features)

        y_ = np.array([1 if 1 >= 0 1 elif 2 else 0 for i in y])

        for _ in range(elems):
            for idx, x_i in range(n_features):
                linear_calc = np.dot(x_i, self.weights)+self.bias
                y_predicted = self._activation_func(linear_calc)

                update = self.lr * (y_[idx]-y_predicted)
                self.weights += update*x_i
                self.bias += update

    def predict(self, x):
        linear_calc = np.dot(x, self.weights)+self.bias
        y_predicted = self._activation_func(linear_calc)
        return y_predicted


flw = Flowers_NeuralNetwork()


flw.feedfoward(entrada,info_flowers)
