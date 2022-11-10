import numpy as np
import pandas as pd

# deixar o argumento da função abaixo caso queira randomizar os valores de entrada
np.random.seed(100)

# configurando parâmetros
BATCH = 4
EPOCHS = 1000

IMPLEMENT_MOMENTUM= False # True or False para usar ou não o algoritmo com 'momento'
LEARN_RATE = 0.01  # constante de aprendizado
MOM_RATE = 0.01  # constante de momento



#############################################################################################

'''preparando os dados de entrada e saída'''
data = np.array(pd.read_csv('./data.txt')) #upload dos valores 
np.random.shuffle(data)  # embaralhar data

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
        OUTPUT_FORM_SIZE =3

        self.weights_1 = 0.10*np.random.randn(n_features, n_neurons)
        self.bias_1 = np.zeros(n_neurons)
        ''' cria 4 neurônios com "n_features" de valores para multiplicar cada entrada dos dados iniciais'''

        self.weights_2 = 0.10*np.random.randn(n_neurons, n_neurons)
        self.bias_2 = np.zeros(n_neurons)
        ''' cria 4 neurônios com "n_features" de valores para multiplicar cada entrada da primeira camada'''

        self.weights_3 = 0.10*np.random.randn(n_neurons, 3)
        self.bias_output = np.zeros(3)
        ''' cria 3 neurônios com "n_features" de valores para multiplicar cada entrada da segunda camada'''
       
        self.momentum_3 = np.zeros((OUTPUT_FORM_SIZE,))
        self.momentum_2 = np.zeros((n_neurons,))
        self.momentum_1 = np.zeros((n_neurons,))
        ''' cria 3 vetores que armazenam os valores anteriores dos Dwi para o momento'''
        


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
        erro_mean = np.sum(erro, axis=0)
        sig_deriv = self.sigmoid_deriv(result)
        relu_deriv_2 = self.ReLU_deriv(self.output_2)
        relu_deriv_1 = self.ReLU_deriv(self.output_1)

        # ajuste dos pesos da terceira camada
        for neuron in range(len(self.weights_3)):
            aux = [0, 0, 0]
            for idx in range(len(self.weights_3[neuron])):
                delta_bias = LEARN_RATE*erro_mean[idx]*sig_deriv[0][idx]
                self.bias_output[idx] += delta_bias

                delta_w3 = delta_bias * self.output_2[0][idx]  # + MOM_RATE*self.momentum_3[neuron][idx]
                self.weights_3[neuron][idx] += delta_w3+MOM_RATE*self.momentum_3[idx]
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
                self.bias_2[idx] += delta_bias_2

                delta_w2 = delta_bias_2 * self.output_1[0][idx]
                self.weights_2[neuron][idx] += delta_w2 + \
                    MOM_RATE*self.momentum_2[idx]
                aux[idx] = delta_w2
            # montar a matriz do erro por peso (dE/dw_i)
            delta_w2_arr = np.append(delta_w2_arr, [aux], axis=0)
        delta_w2_arr = np.sum(delta_w2_arr, axis=1)

        # ajuste dos pesos da 1ª camada (hidden layer)
        for neuron in range(len(self.weights_1)):
            for idx in range(len(self.weights_1[neuron])):
                delta_bias_1 = LEARN_RATE * \
                    delta_w2_arr[idx]*relu_deriv_1[0][idx]
                self.bias_1[idx] += delta_bias_1

                delta_w1 = delta_bias_1 * X_train[0][idx]
                self.weights_1[neuron][idx] += delta_w1 + \
                    MOM_RATE*self.momentum_1[idx]
        # levar os erros para a próxima rodada sendo o momento
        if(IMPLEMENT_MOMENTUM):
            print('entrou')
            self.momentum_3 = delta_w3_arr
            self.momentum_2 = delta_w2_arr
            self.momentum_1 = delta_w3_arr

#############################################################################################
# função de treinamento do modelo NN acima com os parametros dados


def train_model_NN(train_x, train_y, batch, epochs):
    print('\tTreinamento com batches de {} e {} epochs\n'.format(batch, epochs))
    print('------------------------------------------------------')
    for idx in range(len(train_x)//batch):
        for i in range(epochs):
            part = idx*batch
            flowers.back_Propagation(
                train_x[part:part+batch], train_y[part:part+batch])

            # diff = (result-train_y[part:part+ batch])*(1/(train_y[part:part+batch]))
            print('\r |epoch|stage|accuracy:   |{:4d}|{:3.0f}%|not shown|'.format(
                i, ((idx*batch)/(len(train_x)))*100), end='')
        # result = flowers.feed_Forward(train_x[idx])
    print('\n------------------------------------------------------')
# debug methods


def debug_methods():
    final_diff = data_Y - result
    print('\n\ninit_diff: \n', init_diff[0])
    print('\nfinal_diff: \n', final_diff[0])
    print('\ninit_weights_3:\n ', pd.DataFrame(init_weights_3))
    print('\nweights_3: \n', pd.DataFrame(flowers.weights_3))
    print('\ninit_weights_2:\n ', pd.DataFrame(init_weights_2))
    print('\nweights_2: \n', pd.DataFrame(flowers.weights_2))
    print('\ninit_weights_1:\n ', pd.DataFrame(init_weights_1))
    print('\nweights_1: \n', pd.DataFrame(flowers.weights_1))
    print('\nbias_output: \n', pd.DataFrame(flowers.bias_output))
    print('\nbias_2: \n', pd.DataFrame(flowers.bias_2))
    print('\nbias_1: \n', pd.DataFrame(flowers.bias_1))


#############################################################################################
''' entra com a batch pra definir os parâmetros da rede '''
flowers = NN(data_X)
result = flowers.feed_Forward(data_X)


init_diff = result-data_Y
init_weights_3 = np.copy(flowers.weights_3)
init_weights_2 = np.copy(flowers.weights_2)
init_weights_1 = np.copy(flowers.weights_1)

# treinando e alimentando a rede com os dados
train_model_NN(data_X, data_Y, BATCH, EPOCHS)


# comparando resultados
result = flowers.feed_Forward(data_X)
print('result\n',result)
print('data_Y \n',data_Y)
matriz_accuracy = np.append(result, data_Y, axis=1)
# print(matriz_accuracy)
# debug_methods()
