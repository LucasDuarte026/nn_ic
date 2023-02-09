import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

voltage_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-21180.dat', sep="\s+")
velo_fake = pd.read_csv(
    '/mnt/c/Users/micro/OneDrive/Documentos/Faculdade/ic_neural_network/nn_ic/air_analysis/data/hotfilm-fake-vel-21180.dat', sep="\s+")
data = voltage_fake.assign(velocity=velo_fake['velocity'])
print(data)

X = np.array(data.velocity)

print(f'antes X: {X}')

X = np.reshape(X,(len(X),1))

# printing out result
print(f'depois X: {X}')