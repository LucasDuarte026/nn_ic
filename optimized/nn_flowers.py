# Bibliotecas para manipulação de dados
import numpy as np
import pandas as pd
# Bibliotecas para tratamento dos dados em Rede Neural
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Deixar o argumento da função abaixo vazio caso queira randomizar os valores de entrada. Escolhendo-o, mantem o mesmo ponto de partida da regressão
np.random.seed(100)

# Configurando parâmetros
EPOCHS = 100
LEARN_RATE = 0.15  # constante de aprendizado
VAL_PRINT = True  # controle de print do testes de validação


#############################################################################################

# Preparando os dados de entrada e saída

iris = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/all/flowers/projeto_flores/simple/iris.csv')
iris = iris.sample(frac=1).reset_index(drop=True)

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)

one_hot_encoder = OneHotEncoder(sparse=False)

Y = iris.Species

Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1)

#############################################################################################

# Funções necessárias

# cria a rede neural desejada com o os dados de entrada, saída, e os parâmetros de treinamento


def NeuralNetwork(X_train, Y_train, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15):
    weights = InitializeWeights(nodes)

    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        if (epoch % 20 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(
                Accuracy(X_train, Y_train, weights)))
            if X_val.any() and VAL_PRINT == True:
                print("Validation Accuracy:{}".format(
                    Accuracy(X_val, Y_val, weights)))

    return weights

    # Inicializa os pesos com valores entre [-1, 1] (incluindo o bias)


def InitializeWeights(nodes):
    layers, weights = len(nodes), []

    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
             for j in range(nodes[i])]
        weights.append(np.matrix(w))

    return weights


def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation)  # Augment with bias

    return activations


def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal)  # Error at output

    for j in range(layers, 0, -1):
        currActivation = activations[j]
        if (j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]

        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1)  # Remove bias from weights
        error = np.dot(delta, w)  # Calculate error for current layer

    return weights


def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x))  # Augment feature vector

        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights


def reluDerivative(x):
    return np.array([reluDerivativeSingleElement(xi) for xi in x])


def reluDerivativeSingleElement(xi):
    if xi > 0:
        return 1
    elif xi <= 0:
        return 0


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SigmoidDerivative(x):
    return np.multiply(x, 1-x)


def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item)  # Augment feature vector

    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)

    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y  # Return prediction vector


def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if (output[i] > m):
            m, index = output[i], i

    return index


def Accuracy(X, Y, weights):
    """Run set through network, find overall accuracy"""
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = Predict(x, weights)

        if (y == guess):
            # Guessed correctly
            correct += 1

    return correct / len(X)


features = len(X[0])  # Número de "features" (4 para o caso atual)
# Número da saida (como é one hot encoding, são 3 nós "001")
outputs = len(Y[0])

# Número de nós por camada ( camada oculta são os números centrais que podem ser ampliados)
layers = [features, 2, 2, 2, outputs]
lr, epochs = LEARN_RATE, EPOCHS

weights = NeuralNetwork(X_train, Y_train, X_val, Y_val,
                        epochs=epochs, nodes=layers, lr=lr)

print("\nTesting Accuracy: {}".format(Accuracy(X_test, Y_test, weights)))

print("\n\n\nmatriz:\n{}".format(weights))
