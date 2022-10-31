import numpy as np
import pandas as pd

# deixar o argumento da função abaixo caso queira randomizar os valores de entrada
np.random.seed(200)
BATCH = 2
EPOCHS = 1000
LEARN_RATE = 0.1  # constante de aprendizado
data = pd.read_csv('./data.txt')


#############################################################################################
'''preparando os dados de entrada e saída'''

data_X = np.array(data).T[:4].T.astype(float)
data_Y_text = np.array(pd.DataFrame(data).T[4:].T)


''' train_Y gerado a partir do banco de dados  '''
data_Y = np.empty((0, 3)).astype(int)
for i in range(len(data_Y_text)):
    if data_Y_text[i] == 'Iris-setosa':
        data_Y = np.append(data_Y, [[1, 0, 0]], axis=0)
    elif data_Y_text[i] == 'Iris-versicolor':
        data_Y = np.append(data_Y, [[0, 1, 0]], axis=0)
    elif data_Y_text[i] == 'Iris-virginica':
        data_Y = np.append(data_Y, [[0, 0, 1]], axis=0)

#############################################################################################

''' rede neural usada '''


class NN():

    def __init__(self, train_model):
        batches, n_features = np.shape(train_model)
        n_neurons = 4

        self.weights = 0.10*np.random.randn(n_features, n_neurons)
        self.bias_1 = np.zeros(n_neurons)
        ''' cria 4 neurônios com "n_features" de valores para multiplicar cada entrada dos dados iniciais'''

        self.weights_2 = 0.10*np.random.randn(n_neurons, n_neurons)
        self.bias_2 = np.zeros(n_neurons)
        ''' cria 4 neurônios com "n_features" de valores para multiplicar cada entrada da primeira camada'''

        self.weights_3 = 0.10*np.random.randn(n_neurons, 3)
        self.bias_output = np.zeros(3)
        ''' cria 3 neurônios com "n_features" de valores para multiplicar cada entrada da segunda camada'''

    def ReLU(self, input):
        return np.maximum(0, input)

    def ReLU_deriv(self, input):
        return np.maximum(0, )

    def softmax(self, input):     # não usada
        output_softmax = np.exp(input - np.max(input, axis=1, keepdims=True))
        normalized = output_softmax / \
            np.sum(output_softmax, axis=1, keepdims=True)
        return normalized

    def sigmoid(self, value):
        return 1 / (1 + np.exp(- value))

    def sigmoid_deriv(self, value):
        a = self.sigmoid(value)
        return a*(1-a)

    def feed_Forward(self, X_train):
        self.output_1 = self.ReLU(np.dot(X_train, self.weights) + self.bias_1)
        self.output_2 = self.ReLU(
            np.dot(self.output_1, self.weights_2) + self.bias_2)
        result = self.sigmoid(
            np.dot(self.output_2, self.weights_3) + self.bias_output)
        return result

    def error(self, X_train, Y_train):
        Y_loss = self.feed_Forward(X_train)
        loss = np.multiply(0.5, np.square(Y_train - Y_loss))
        error = np.sum(loss, axis=0)
        return error

    def back_Propagation(self, X_train, Y_train):
        result = self.feed_Forward(X_train)
        erro = result - Y_train
        for neuron in range(len(self.weights_3)):
            for idx in range(len(self.weights_3[neuron])):
                delta_w = LEARN_RATE*erro[0][idx]
                self.bias_output[idx] = delta_w

                delta_w *= self.output_2[0][idx]
                self.weights_3[neuron][idx] -= delta_w


#############################################################################################

''' entra com a batch pra definir os parâmetros da rede '''
flowers = NN(data_X)
result = flowers.feed_Forward(data_X)


# debug methods

# print("weights_1\n", pd.DataFrame(flowers.weights))
# print("weights_2\n", pd.DataFrame(flowers.weights_2))
# print("\nweights_3\n", pd.DataFrame(flowers.weights_3))
# print("output_1\n", pd.DataFrame(flowers.output_1))
# print("\noutput_2\n", pd.DataFrame(flowers.output_2))
# print("\nbias_output\n", pd.DataFrame(flowers.bias_output))
# print("\nresult\n", pd.DataFrame(result))
# print("\ndata\n", pd.DataFrame(data_Y))

init_diff = result-data_Y

''' treinando e alimentando a rede com os dados '''
for i in range(EPOCHS):
    for idx in range(len(data_X)//BATCH):
        part = idx*BATCH
        print('\r epoch | batch:{:4d}|{:2d}'.format(i, idx), end='')
        flowers.back_Propagation(
            data_X[part:part+BATCH], data_Y[part:part+BATCH])
        result = flowers.feed_Forward(data_X[part:part+BATCH])
        diff = result-data_Y[part:part+BATCH]

print()
result = flowers.feed_Forward(data_X)
final_diff = result-data_Y
print('init_diff: ', np.mean(init_diff, axis=0))
print('final_diff: ', np.mean(final_diff, axis=0))
print('weights: ', pd.DataFrame(flowers.weights_3))
print('bias_output: ', pd.DataFrame(flowers.bias_output))
