import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
print("\n\n\n-- -- -- -- -- -- data_insert -- -- -- -- -- -- \n\n\n")
voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(velocity=velo_fake['velocity'])
print(data)
X = data.voltage
Y = data.velocity/100

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

print(X)
print(X[0])
