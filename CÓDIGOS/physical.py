import numpy as np
import pandas as pd
from tqdm import tqdm

import library

volunteers = np.array([[1.88, 90.0], [1.80, 89.0], [1.78, 62.0], [1.70, 77.0], [1.80, 60.0], [1.75, 70.0]])


def physical_parameters(height, weight):
    return [0.028*weight, 0.016*weight, 0.081096*height, 0.06278*height, 0.0001004374*(height**2)*weight,
            0.000031312*(height**2)*weight, 0.186*height, 0.146*height]


"""o vetor 'kinematics' contém informações organizadas em Angle_Shoulder[:,0], Angle_Elbow[:,1], Velocity_Shoulder[:,2], 
Velocity_Elbow[:,3], Acceleration_Shoulder[:,4] e Acceleration_Elbow[:,5]. Já o vetor 'person_data' contém informações 
estimadas dos valores de momento de inércia, comprimento e massa dos membros superiores dos voluntários, organizados na
seguinte ordem: massa do braço[0] e do antebraço[1], distância das juntas para o centro de massa [2][3], momento de 
inércia[4][5] e comprimento[6][7] de cada membro, ou seja, é um vetor com formato(1,8)"""


def torque(data, person_data):
    data['NT_Elbow'] = (person_data[5] + person_data[1]*(person_data[3]**2))*data['Acceleration_Elbow']
    data['IT_Elbow'] = (-(person_data[5]+person_data[1]*person_data[3]*person_data[6]*np.cos(data['Angle_Shoulder']))*data['Acceleration_Shoulder']
                - (person_data[1]*person_data[3]*person_data[6]*np.sin(data['Angle_Shoulder']))*data['Velocity_Shoulder'])
    data['MT_Elbow'] = data['NT_Elbow'] - data['IT_Elbow'] #aplicar o filtro aqui
    data['NT_Shoulder'] = (person_data[4]+person_data[0]*(person_data[2]**2))*data['Acceleration_Shoulder']
    data['IT_Shoulder'] = (-(person_data[5]+person_data[1]*(person_data[6]**2+person_data[3]**2 +
                    2*person_data[6]*person_data[3]*np.cos(data['Angle_Shoulder'])))*data['Acceleration_Elbow'] -
                   (person_data[5]+person_data[1]*person_data[1]*person_data[3]*person_data[6]*np.cos(data['Angle_Shoulder']))*
                    data['Acceleration_Shoulder'] + person_data[1]*person_data[3]*person_data[6]*np.sin(data['Angle_Shoulder'])*data['Acceleration_Shoulder']
                   + 2*person_data[1]*person_data[3]*person_data[6]*np.sin(data['Angle_Shoulder'])*data['Velocity_Shoulder']*data['Velocity_Elbow'])
    data['MT_Shoulder'] = data['NT_Shoulder'] - data['IT_Shoulder']     #aplicar o filtro aqui
    data.drop(['NT_Elbow', 'IT_Elbow', 'NT_Shoulder', 'IT_Shoulder'], axis=1, inplace=True)
    return data


"""def write_torque_data():
    arq = []
    vol = "vol10"
    freq = ["freq02", "freq03"]
    amp = ["amp01", "amp02"]
    trial = ["trial01", "trial02", "trial03", "trial04"]
    vol_parameters = physical_parameters(1.75, 70.0)
    for j in range(len(freq)):
        for k in range(len(amp)):
            for t in range(len(trial)):
                arq.append(vol + "_" + freq[j] + "_" + amp[k] + "_" + trial[t])
    for i in range(len(arq)):
        data1 = pd.read_csv("..//SYNCHRONIZED DATA//" + arq[i] + ".csv")
        data1 = torque(data1, vol_parameters)
        data1.to_csv("..//SYNCHRONIZED DATA//TORQUE DATA/" + arq[i] + ".csv", index=False)
    return"""


def write_torque_data():
    vol = ['vol01', 'vol03', 'vol04', 'vol05', 'vol08', 'vol10']
    freq = ['freq02', 'freq03']
    amp = ['amp01', 'amp02']
    trial = ["trial01", "trial02", "trial03", "trial04"]
    arq = []
    for i in vol:
        for j in freq:
            for k in amp:
                for t in trial:
                    arq.append(i + "_" + j + "_" + k + "_" + t)
    for name in tqdm(arq):
        if name.split('_')[0] == 'vol01':
            parameters = physical_parameters(1.88, 90)
        elif name.split('_')[0] == 'vol03':
            parameters = physical_parameters(1.80, 89)
        elif name.split('_')[0] == 'vol04':
            parameters = physical_parameters(1.78, 62)
        elif name.split('_')[0] == 'vol05':
            parameters = physical_parameters(1.7, 77)
        elif name.split('_')[0] == 'vol08':
            parameters = physical_parameters(1.8, 60)
        else:
            parameters = physical_parameters(1.75, 70)
        data1 = pd.read_csv("..//SYNCHRONIZED DATA//FILTERED DATA/" + name + ".csv")
        data1[['Angle_Shoulder', 'Angle_Elbow', 'Velocity_Shoulder', 'Velocity_Elbow', 'Acceleration_Shoulder',
               'Acceleration_Elbow']] = data1[['Angle_Shoulder', 'Angle_Elbow', 'Velocity_Shoulder', 'Velocity_Elbow',
                                              'Acceleration_Shoulder', 'Acceleration_Elbow']].apply(lambda x: x*np.pi/180)
        data1 = torque(data1, parameters)
        data1.to_csv("..//SYNCHRONIZED DATA//TORQUE DATA/" + name + ".csv", index=False)
    return


def filtered_data():
    vol = ['vol01', 'vol03', 'vol04', 'vol05', 'vol08', 'vol10']
    freq = ['freq02', 'freq03']
    amp = ['amp01', 'amp02']
    trial = ["trial01", "trial02", "trial03", "trial04"]
    emg = ['EMG_PC', 'EMG_DA', 'EMG_DP', 'EMG_BI', 'EMG_TR', 'EMG_BR']
    arq = []
    for i in vol:
        for j in freq:
            for k in amp:
                for t in trial:
                    arq.append(i + "_" + j + "_" + k + "_" + t)
    for name in tqdm(arq):
        data = pd.read_csv("..//SYNCHRONIZED DATA//DATA/" + name + ".csv")
        for muscle in emg:
            data[muscle] = library.butterwoth_lowpass(data[muscle], 4, 30)
        data.to_csv("..//SYNCHRONIZED DATA//FILTERED DATA/" + name + ".csv", index=False)
    return
