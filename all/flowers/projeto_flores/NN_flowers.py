import numpy as np
import pandas as pd

# deixar o argumento da função abaixo caso queira randomizar os valores de entrada
np.random.seed(100)
BATCH = 1
EPOCHS = 100
LEARN_RATE = 0.01 # constante de aprendizado
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

        self.weights_1 = 0.10*np.random.randn(n_features, n_neurons)
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
        aux = input
        for i in range(len(input)):
            for j in range(len(input[i])):
                if input[i][j] <= 0:
                    aux[i][j] = 0
                elif input[i][j] > 0:
                    aux[i][j] = 1
        return np.maximum(0, aux)

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
        self.output_1 = self.ReLU(
            np.dot(X_train, self.weights_1) + self.bias_1)
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

    # treina os pesos com o banco de dados x_train e o resultado desejado Y_train
    def back_Propagation(self, X_train, Y_train):

        # ajuste dos pesos e bias da saída/output
        delta_w3_arr = np.empty((0, 3))
        delta_w2_arr = np.empty((0, 4))

        # parametros gerais que serão utilizados por rodada de back propagation
        result = self.feed_Forward(X_train)
        erro = Y_train-result
        sig_deriv = self.sigmoid_deriv(result)
        relu_deriv_2 = self.ReLU_deriv(self.output_2)
        relu_deriv_1 = self.ReLU_deriv(self.output_1)

        # ajuste dos pesos da terceira camada
        for neuron in range(len(self.weights_3)):
            aux = [0, 0, 0]
            for idx in range(len(self.weights_3[neuron])):
                delta_bias = LEARN_RATE*erro[0][idx]*sig_deriv[0][idx]
                self.bias_output[idx] -= delta_bias

                delta_w3 = delta_bias * self.output_2[0][idx]
                self.weights_3[neuron][idx] -= delta_w3
                aux[idx] = delta_w3
            # montar a matriz do erro por peso (dE/dw_i)
            delta_w3_arr = np.append(delta_w3_arr, [aux], axis=0)

        # Erro do neurônio pela soma dos dE/dw_i
        delta_w3_arr = np.sum(delta_w3_arr, axis=1)

        # ajuste dos pesos da segunda camada (hidden layer)
        for neuron in range(len(self.weights_2)):
            aux = [0, 0, 0, 0]
            for idx in range(len(self.weights_2[neuron])):
                delta_bias_2 = LEARN_RATE * \
                    delta_w3_arr[idx] * relu_deriv_2[0][idx]
                self.bias_2[idx] -= delta_bias_2

                delta_w2 = delta_bias_2 * self.output_1[0][idx]
                self.weights_2[neuron][idx] -= delta_w2
                aux[idx] = delta_w2
            # montar a matriz do erro por peso (dE/dw_i)
            delta_w2_arr = np.append(delta_w2_arr, [aux], axis=0)
        delta_w2_arr = np.sum(delta_w2_arr, axis=1)
        # print('\n',delta_w2_arr)

        # ajuste dos pesos da 1ª camada (hidden layer)
        for neuron in range(len(self.weights_1)):
            for idx in range(len(self.weights_1[neuron])):
                delta_bias_1 = LEARN_RATE * \
                    delta_w2_arr[idx]*relu_deriv_1[0][idx]
                self.bias_1[idx] -= delta_bias_1

                delta_w1 = delta_bias_1 * X_train[0][idx]
                self.weights_1[neuron][idx] -= delta_w1


#############################################################################################
# função de treinamento do modelo NN acima com os parametros dados
def train_model_NN(train_x, train_y, batch, epochs):
    for i in range(epochs):
        for idx in range(len(train_x)//batch):
            part = idx*batch
            flowers.back_Propagation(
                train_x[part:part+batch], train_y[part:part+batch])
            # diff = (result-train_y[part:part+batch])*(1/(train_y[part:part+batch]))
            print('\r epoch | batch | accuracy:{:4d}|{:3d}| not shown.'.format(
                i, idx,), end='')


#############################################################################################
''' entra com a batch pra definir os parâmetros da rede '''
flowers = NN(data_X)
result = flowers.feed_Forward(data_X)




init_diff = result-data_Y
init_weights_3 = flowers.weights_3
init_weights_2 = flowers.weights_2
init_weights_1 = flowers.weights_1

''' treinando e alimentando a rede com os dados '''

# train_model_NN(data_X, data_Y, BATCH, EPOCHS)
for idx in range(EPOCHS):
    for i in range(130):
        flowers.back_Propagation(data_X[i:i+1], data_Y[i:i+1])
        acc= data_Y[i:i+1]-flowers.feed_Forward(data_X[i:i+1])
        print('\r |{:4d}| |{:4d}| |acc: |{:5f}|{:5f}|{:5f}\n'.format(i, idx,acc[0][0],acc[0][1],acc[0][2]), end='')
# comparando resultados

result = flowers.feed_Forward(data_X)
final_diff = result-data_Y

# debug methods

print('\n\ninit_diff: \n', np.mean(init_diff, axis=0))
print('\nfinal_diff: \n', np.mean(final_diff, axis=0))
print('\ninit_weights_3:\n ', pd.DataFrame(init_weights_3))
print('\nweights_3: \n', pd.DataFrame(flowers.weights_3))
print('\ninit_weights_2:\n ', pd.DataFrame(init_weights_2))
print('\nweights_2: \n', pd.DataFrame(flowers.weights_2))
print('\ninit_weights_1:\n ', pd.DataFrame(init_weights_1))
print('\nweights_1: \n', pd.DataFrame(flowers.weights_1))
print('\nbias_output: \n', pd.DataFrame(flowers.bias_output))
print('\nbias_2: \n', pd.DataFrame(flowers.bias_2))
print('\nbias_1: \n', pd.DataFrame(flowers.bias_1))
