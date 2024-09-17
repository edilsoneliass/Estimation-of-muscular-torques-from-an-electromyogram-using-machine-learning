import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scalers
import model
from tqdm import tqdm

"""Todos os vetores devem ser numpy arrays"""
"""Normalized Root Mean Squared Error (NRMSE)"""


def nrmse(y_tgt, y_pred):
    return np.mean(np.sqrt((y_tgt - y_pred) ** 2)) / np.std(y_tgt)


"""Slope"""


def slope(y_tgt, y_pred):
    y = []
    for i in range(y_tgt.shape[0]):
        y.append([y_tgt[i]])
    model = LinearRegression()
    model.fit(y, y_pred)
    return model.coef_[0]


"""Correlation"""


def correlation(y_tgt, y_pred):
    return np.corrcoef(y_tgt, y_pred)[0, 1]


def statistics_from_data(y_pred, y_tgt):
    return [nrmse(y_tgt['0'], y_pred[:, 0]), nrmse(y_tgt['1'], y_pred[:, 1]),
            slope(y_tgt['0'], y_pred[:, 0]), slope(y_tgt['1'], y_pred[:, 1]),
            correlation(y_tgt['0'], y_pred[:, 0]), correlation(y_tgt['1'], y_pred[:, 1])]



    """directory = '..//NEURAL NETWORKS//MODELOS//LAYERS3_NEURONS256'
    vol = ['vol01', 'vol03', 'vol04', 'vol05', 'vol08', 'vol10']
    freq = ['freq02', 'freq03']
    amp = ['amp01', 'amp02']
    vfa = []
    mse_elbow = []
    slp_elbow = []
    corr_elbow = []
    mse_shoulder = []
    slp_shoulder = []
    corr_shoulder = []
    for v in tqdm(vol):
        for f in freq:
            for a in amp:
                scaler_x = scalers.load_scaler(v + '_' + f + '_' + a + '_scaler_x')
                scaler_y = scalers.load_scaler(v + '_' + f + '_' + a + '_scaler_y')
                modelo = model.load(v + '_' + f + '_' + a, directory)
                x = model.input_test_data([v + '_' + f + '_' + a + '_trial04_features.csv'], scaler_x)
                y = model.predict(modelo, x, scaler_y)
                y_test = model.output_test_data([v + '_' + f + '_' + a + '_trial04_output.csv'])
                vfa.append(v + '_' + f + '_' + a)
                mse_elbow.append(nrmse(y_test['0'], y[:, 0]))
                slp_elbow.append(slope(y_test['0'], y[:, 0]))
                corr_elbow.append(correlation(y_test['0'], y[:, 0]))
                mse_shoulder.append(nrmse(y_test['1'], y[:, 1]))
                slp_shoulder.append(slope(y_test['1'], y[:, 1]))
                corr_shoulder.append(correlation(y_test['1'], y[:, 1]))
    data = pd.DataFrame(mse_elbow, columns=['NRMSE_Elbow'])
    data['NRMSE_Shoulder'] = slp_shoulder
    data['Slope_Elbow'] = slp_elbow
    data['Slope_Shoulder'] = slp_shoulder
    data['Correlation_Elbow'] = corr_elbow
    data['Correlation_Shoulder'] = corr_shoulder
    data.index = vfa
    describe = pd.DataFrame(data.describe())
    data = pd.concat([data, describe], axis=0)
    data.to_csv(directory + '//' + 'statistics.csv', index=True)
    print(data.head())
    print(describe)"""
