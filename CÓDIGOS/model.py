import keras
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import library
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model


"""Remember that layer_shape is a vector"""


def build_model(input_shape, layers_shape, activation, output_shape, loss, metrics):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    model = models.Sequential()
    model.add(layers.Dense(layers_shape[0], activation[0], input_shape=input_shape, kernel_initializer=initializer))
    for i in range(1, len(layers_shape)):
        model.add(layers.Dense(layers_shape[i], activation[i]))
    model.add(layers.Dense(output_shape, activation='linear', kernel_initializer=initializer))
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=0.0000025), metrics=metrics)
    return model


def save_model(model, name, directory):
    model.save(directory + '//' + name + '.h5')
    return


def load(name, directory):
    return load_model(directory + '//' + name + '.h5')


def write_feature_data(data, timestep, name):
    x = []
    for emg in tqdm(np.squeeze(
            sliding_window_view(data[['EMG_PC', 'EMG_DA', 'EMG_DP', 'EMG_BI', 'EMG_TR', 'EMG_BR']].to_numpy(),
                                (150, 6))[::50, :], axis=1)):
        x_ = []
        for i in range(emg.shape[1]):
            x_.append(np.squeeze(library.features(emg[:, i], timestep)))
        x.append(np.array(x_).flatten())
    x = np.array(x)
    x = pd.DataFrame(x)
    x.to_csv("..//SYNCHRONIZED DATA//FEATURES DATA/" + name + "_features.csv", index=False)
    return


def write_output_data(data, name):
    y = []
    for torque in tqdm(np.squeeze(
            sliding_window_view(data[['MT_Elbow', 'MT_Shoulder']].to_numpy(),
                                (150, 2))[::50, :], axis=1)):
        y.append(np.array([np.mean(torque[:, 0]), np.mean(torque[:, 1])]))
    y = pd.DataFrame(np.array(y))
    y.to_csv("..//SYNCHRONIZED DATA//OUTPUT DATA/" + name + "_output.csv", index=False)
    return


def export_data():
    arq = []
    vol = ['vol10']                                                                                 #'vol10']
    freq = ['freq02', 'freq03']
    amp = ['amp01', 'amp02']
    trial = ["trial01", "trial02", "trial03", "trial04"]
    for i in range(len(vol)):
        for j in range(len(freq)):
            for k in range(len(amp)):
                for t in range(len(trial)):
                    arq.append(vol[i] + "_" + freq[j] + "_" + amp[k] + "_" + trial[t])
    for name in tqdm(arq):
        data = pd.read_csv("..//SYNCHRONIZED DATA//TORQUE DATA//" + name + ".csv")
        write_feature_data(data, 0.001, name)
        write_output_data(data, name)
    return


def input_train_data(archives):
    data = pd.read_csv("..//SYNCHRONIZED DATA//FEATURES DATA//" + archives[0])
    scaler = StandardScaler()
    for i in range(1, len(archives)):
        df = pd.read_csv("..//SYNCHRONIZED DATA//FEATURES DATA//" + archives[i])
        data = pd.concat([data, df], axis=0)
    data = pd.DataFrame(scaler.fit_transform(data))
    return data, scaler


def output_train_data(archives):
    data = pd.read_csv("..//SYNCHRONIZED DATA//OUTPUT DATA//" + archives[0])
    scaler = StandardScaler()
    for i in range(1, len(archives)):
        df = pd.read_csv("..//SYNCHRONIZED DATA//OUTPUT DATA//" + archives[i])
        data = pd.concat([data, df], axis=0)
    data = pd.DataFrame(scaler.fit_transform(data))
    return data, scaler


def input_test_data(archives, scaler):
    data = scaler.transform(pd.read_csv("..//SYNCHRONIZED DATA//FEATURES DATA//" + archives[0]))
    return data


def output_test_data(archives):
    data = pd.read_csv("..//SYNCHRONIZED DATA//OUTPUT DATA//" + archives[0])
    return data


def output_validation_data(archives, scaler):
    data = scaler.transform(pd.read_csv("..//SYNCHRONIZED DATA//OUTPUT DATA//" + archives[0]))
    return data


def inverse(signal, scaler):
    return scaler.inverse_transform(signal)


def predict(modelo, data, scaler):
    prediction = pd.DataFrame(modelo.predict(data))
    prediction[0] = library.butterwoth_lowpass(np.array(prediction[0]),4,5,20)
    prediction[1] = library.butterwoth_lowpass(np.array(prediction[1]),4,5,20)
    return scaler.inverse_transform(prediction)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    return


def processing_data(archives):
    archives_input = [archives + '_trial01_features.csv', archives + '_trial02_features.csv']
    archives_output = [archives + '_trial01_output.csv', archives + '_trial02_output.csv']
    x_train, scaler_x = input_train_data(archives_input)
    x_validation = input_test_data([archives + '_trial03_features.csv'], scaler_x)
    x_test = input_test_data([archives + '_trial04_features.csv'], scaler_x)
    y_train, scaler_y = output_train_data(archives_output)
    y_validation = output_validation_data([archives + '_trial03_output.csv'], scaler_y)
    y_test = output_test_data([archives + '_trial04_output.csv'])
    return x_train, y_train, x_validation, y_validation, x_test, y_test, scaler_y


def plot_prediction(y_pred, y_test, directory_figs, archives):
    t = 0.050 * np.arange(len(y_pred[:, 0]))

    pparam1 = dict(xlabel='Time (s)', ylabel=r'Elbow Muscle Torque(Nm)')
    plt.rcParams.update({'figure.dpi': '100'})
    fig, ax = plt.subplots()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.plot(t, y_test['0'], label='Target', color='r')
    ax.plot(t, y_pred[:, 0], label='Predicted', color='b')
    ax.legend(loc='best')
    ax.autoscale(tight=True)
    ax.set(**pparam1)
    fig.savefig(directory_figs + '//' + archives + '_elbow.png', dpi=300)
    plt.show()
    plt.close()

    pparam2 = dict(xlabel='Time (s)', ylabel=r'Shoulder Muscle Torque (Nm)')
    plt.rcParams.update({'figure.dpi': '100'})
    fig, ax = plt.subplots()
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.plot(t, y_test['1'], label='Target', color='r')
    ax.plot(t, y_pred[:, 1], label='Predicted', color='b')
    ax.autoscale(tight=True)
    ax.set(**pparam2)
    ax.legend(loc='best')
    fig.savefig(directory_figs + '//' + archives + '_shoulder.png', dpi=300)
    plt.show()
    plt.close()
    return