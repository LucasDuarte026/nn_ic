import numpy as np
import pandas as pd

# deixar o argumento da função abaixo caso queira randomizar os valores de entrada
np.random.seed(200)
BATCH = 40
EPOCHS = 100
LEARN_RATE = 0.3  # constante de aprendizado
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
data_Y_test = np.array(pd.DataFrame(data_Y)[:10])

#############################################################################################


''' rede neural usada '''
class NN():

    def __init__(self, train_model):
        batches, n_features = np.shape(train_model)
        n_neurons = 4

        self.weights = 0.10*np.random.randn(n_features, n_neurons)
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
        self.output_1 = self.ReLU(np.dot(X_train, self.weights) + self.bias)
        self.output_2 = self.ReLU(
            np.dot(self.output_1, self.weights_2) + self.bias_2)
        result = self.sigmoid(
            np.dot(self.output_2, self.weights_3) + self.bias_3)
        return result

    def error(self, X_train, Y_train):
        Y_loss = self.feed_Forward(X_train)
        loss = np.multiply(0.5, np.square(Y_train - Y_loss))
        error = np.sum(loss, axis=0)
        return error

    def back_Propagation(self, X_train, Y_train):
        error = self.error(X_train, Y_train)
        print("error\n", error)
        
        
#############################################################################################


''' entra com a batch pra definir os parâmetros da rede '''
flowers = NN(data_X[:BATCH])
result = flowers.feed_Forward(data_X[:BATCH])
flowers.back_Propagation(data_X[:BATCH],data_Y[:BATCH])

# print("weights_1\n", pd.DataFrame(flowers.weights))
# print("weights_2\n", pd.DataFrame(flowers.weights_2))
# print("weights_3\n", pd.DataFrame(flowers.weights_3))
# print("output_1\n", pd.DataFrame(flowers.output_1))
# print("output_2\n", pd.DataFrame(flowers.output_2))
print("result\n", pd.DataFrame(result))
print("data\n", pd.DataFrame(data_Y[:BATCH]))


''' treinando e alimentando a rede com os dados '''

new_weights = np.empty((0, flowers.weights_3.shape[1]))

# print("error\n", pd.DataFrame(error))
# print("error\n", pd.DataFrame(error-result))


'''
print("antes:\n", pd.DataFrame(flowers.weights_3))
for idx in range(EPOCHS):
    new_weights = np.empty((0, 3))
    for i in range(len(flowers.weights_3)):
        ins_arr = [0, 0, 0]
        error = flowers.back_Propagation(data_X[:BATCH], data_Y[:BATCH])
        result = flowers.feed_Forward(data_X[:BATCH])
        sig_deriv_arr = flowers.sigmoid_deriv(result)

        for j in range(len(flowers.weights_3[i])):
            new_value = flowers.weights_3[i][j]
            changing = (error[j]-result[i][0]) * flowers.output_2[i][j]*sig_deriv_arr[i][j]
            new_value += LEARN_RATE * changing
            ins_arr[j] = new_value
        new_weights = np.append(new_weights, [ins_arr], axis=0)

    # print("\nnew_weights\n", pd.DataFrame(new_weights))
    flowers.weights_3 = new_weights
    result = flowers.feed_Forward(data_X[:BATCH])
    # print("new_result\n", pd.DataFrame(new_result))
    # print("data_Y[:BATCH]\n", pd.DataFrame(data_Y[:BATCH]))
print("dps:\n", pd.DataFrame(flowers.weights_3))
print("result:\n", pd.DataFrame(result))
'''
