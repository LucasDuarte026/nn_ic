import numpy as np
import pandas as pd

# deixar o argumento da função abaixo caso queira randomizar os valores de entrada
np.random.seed(0)
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
''' batch de 10 elementos para teste '''

data_X = np.array(data).T[:4].T
data_Y_text = np.array(pd.DataFrame(data).T[4:].T)
data_test = np.array(pd.DataFrame(data_test).T[:4].T)
data_X = data_X.astype(float)
data_test = data_test.astype(float)

''' train_Y gerado a partir do banco de dados  '''
data_Y = np.empty((0, 3)).astype(int)
for i in range(len(data_Y_text)):
    if data_Y_text[i] == 'Iris-setosa':
        data_Y = np.append(data_Y, [[1, 0, 0]], axis=0)
    elif data_Y_text[i] == 'Iris-versicolor':
        data_Y = np.append(data_Y, [[0, 1, 0]], axis=0)
    elif data_Y_text[i] == 'Iris-virginica':
        data_Y = np.append(data_Y, [[0, 0, 1]], axis=0)
data_Y_test = np.array(pd.DataFrame(data_Y)[:10])

''' rede neural usada '''


class NN():

    def __init__(self, train_model):
        batches, n_features = np.shape(train_model)
        n_neurons = 4

        self.weights = np.random.randn(n_features, n_neurons)
        self.bias = np.zeros(n_neurons)
        ''' cria 4 neurônios com "n_features" de valores para multiplicar cada entrada dos dados iniciais'''

        self.weights_2 = 0.10*np.random.randn(n_neurons, n_neurons)
        self.bias_2 = np.zeros(n_neurons)
        ''' cria 4 neurônios com "n_features" de valores para multiplicar cada entrada da primeira camada'''

        self.weights_3 = 0.10*np.random.randn(n_neurons, 3)
        self.bias_3 = np.zeros(3)
        ''' cria 3 neurônios com "n_features" de valores para multiplicar cada entrada da segunda camada'''

    def ReLU(self, input):
        return np.maximum(0, input)

    def ReLU_deriv(self, input):
        return np.maximum(0, )

    def softmax(self, input):
        output = np.exp(input - np.max(input, axis=1, keepdims=True))
        normalized = output/np.sum(output, axis=1, keepdims=True)
        return normalized

    def softmax_deriv(self):
        pass

    def feed_Forward(self, X_train):
        output = self.ReLU(np.dot(X_train, self.weights) + self.bias)
        output_2 = self.ReLU(np.dot(output, self.weights_2) + self.bias_2)
        # print("anterior\n",np.dot(output_2, self.weights_3))
        result = self.softmax(np.dot(output_2, self.weights_3) + self.bias_3)
        # print("result\n",result)
        return result

    def loss(self, X_train, Y_train):
        output = self.feed_Forward(X_train)
        # print(output)
        # print(Y_train)
        # incase_value = output[range(len(output)), Y_train]
        # loss = -np.log(np.clip(incase_value, 1E-7, 1-1E-7))
        loss = np.multiply(0.5, np.square(Y_train - output))
        return loss

    def back_Propagation(self, X_train, Y_train):
        output = self.loss(X_train, Y_train)

        return output


''' entra com a batch pra definir os parâmetros da rede '''
flowers = NN(data_X)
result = flowers.feed_Forward(data_X[:10])
# print(result)
# print(data_Y[:10])
# print("weights_3\n",flowers.weights_3)

''' treinando e alimentando a rede com os dados '''

loss = flowers.back_Propagation(data_X[:10], data_Y[:10])
error = np.sum(loss, axis=0)
print("data_X\n", data_X[:10])
print("data_Y\n", data_Y[:10])

# print("loss\n", pd.DataFrame(loss))

print("error\n", pd.DataFrame(error))

