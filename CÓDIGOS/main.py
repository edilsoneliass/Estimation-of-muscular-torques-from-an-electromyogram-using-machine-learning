"""import libraries"""

import os
import keras.metrics
import pandas as pd
import model
import statistics


loss_df = pd.read_csv('..//NEURAL NETWORKS//MODELOS//losses.csv')
vol = ['vol01', 'vol03', 'vol04', 'vol05', 'vol08', 'vol10']
freq = ['freq02', 'freq03']
amp = ['amp01', 'amp02']
models = [[[256, 256, 256, 256], ['relu', 'relu', 'relu', 'relu']]]
index = []
statistics_data = []
for m in models:
    directory_models = '..//NEURAL NETWORKS//MODELOS//LAYERS' + str(len(m[0])) + '_NEURONS' + str(m[0][0])
    directory_figs = '..//NEURAL NETWORKS//RESULTADOS PARCIAIS//LAYERS' + str(len(m[0])) + '_NEURONS' + str(m[0][0])
    os.makedirs(directory_models)
    os.makedirs(directory_figs)
    train_loss = 0
    validation_loss = 0
    train_maer = 0
    validation_maer = 0
    for v in vol:
        for f in freq:
            for a in amp:
                archive = v + '_' + f + '_' + a
                x_train, y_train, x_validation, y_validation, x_test, y_test, scaler_y = model.processing_data(archive)
                modelo = model.build_model((360,), m[0], m[1], 2, 'mean_squared_error',
                                           [keras.metrics.MeanAbsolutePercentageError()])
                modelo.summary()
                history = modelo.fit(x_train, y_train, batch_size=1, epochs=256, validation_data=(x_validation, y_validation))
                train_loss += history.history['loss'][-1]
                validation_loss += history.history['val_loss'][-1]
                train_maer += history.history['mean_absolute_percentage_error'][-1]
                validation_maer += history.history['val_mean_absolute_percentage_error'][-1]

                y = model.predict(modelo, x_test, scaler_y)
                model.plot_prediction(y, y_test, directory_figs, archive)
                model.save_model(modelo, archive, directory_models)
                statistics_data.append(statistics.statistics_from_data(y, y_test))
                index.append(archive)
    data = pd.DataFrame(statistics_data, columns=['NRMSE_Elbow', 'NRMSE_Shoulder', 'Slope_Elbow', 'Slope_Shoulder',
                                                  'Correlation_Elbow', 'Correlation_Shoulder'])
    data.index = index
    describe = pd.DataFrame(data.describe())
    data = pd.concat([data, describe], axis=0)
    data.to_csv(directory_models + '//' + 'statistics.csv', index=True)
    id = 'LAYERS' + str(len(m[0])) + '_NEURONS' + str(m[0][0])
    df = pd.DataFrame([[id, train_loss, validation_loss, train_maer, validation_maer]], columns=['Model', 'Train_Loss',
                                                                                                 'Validation_Loss',
                                                                                                 'Train_Mean_Absolute_Percentage_Error',
                                                                                                 'Validation_Mean_Absolute_Percentage_Error'])

    loss_df = pd.concat([loss_df, df], axis=0)
    loss_df.to_csv('..//NEURAL NETWORKS//MODELOS//losses.csv', index=False)
