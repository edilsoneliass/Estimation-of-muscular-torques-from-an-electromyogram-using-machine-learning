from pickle import dump, load
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def input_scaler(archives):
    data = pd.read_csv("..//SYNCHRONIZED DATA//FEATURES DATA//" + archives[0])
    scaler = StandardScaler()
    for i in range(1, len(archives)):
        df = pd.read_csv("..//SYNCHRONIZED DATA//FEATURES DATA//" + archives[i])
        data = pd.concat([data, df], axis=0)
    scaler.fit_transform(data)
    return scaler


def output_scaler(archives):
    data = pd.read_csv("..//SYNCHRONIZED DATA//OUTPUT DATA//" + archives[0])
    scaler = StandardScaler()
    for i in range(1, len(archives)):
        df = pd.read_csv("..//SYNCHRONIZED DATA//OUTPUT DATA//" + archives[i])
        data = pd.concat([data, df], axis=0)
    scaler.fit_transform(data)
    return scaler


def save_scaler(scaler, name):
    dump(scaler, open('..//NEURAL NETWORKS//SCALERS//' + name + '.pkl', 'wb'))
    return


def load_scaler(name):
    return load(open('..//NEURAL NETWORKS//SCALERS//' + name + '.pkl', 'rb'))


"""vol = ['vol01', 'vol03', 'vol04', 'vol05', 'vol08', 'vol10']
freq = ['freq02', 'freq03']
amp = ['amp01', 'amp02']
for v in tqdm(vol):
    for f in freq:
        for a in amp:
            archives_input = [v + '_' + f + '_' + a + '_trial01_features.csv',
                              v + '_' + f + '_' + a + '_trial02_features.csv']
            archives_output = [v + '_' + f + '_' + a + '_trial01_output.csv',
                               v + '_' + f + '_' + a + '_trial02_output.csv']
            scaler_x = input_scaler(archives_input)
            scaler_y = output_scaler(archives_output)
            save_scaler(scaler_x, v + '_' + f + '_' + a + '_scaler_x')
            save_scaler(scaler_y, v + '_' + f + '_' + a + '_scaler_y')"""
