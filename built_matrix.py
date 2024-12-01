import pandas as pd
import numpy as np

file_list = ['gym/Purdue_Recreation_&_Wellness-2024-02-19_23-10-44',
             'gym/Purdue_Recreation_&_Wellness-2024-04-07_23-39-52',
             'home/3141_Swindon_Way-2024-03-18_03-32-12',
             'home/3141_Swindon_Way-2024-04-11_17-09-56',
             'home/3141_Swindon_Way-2024-04-11_17-09-56',
             'lab/Purdue_University-2024-02-20_01-44-19',
             'lab/Purdue_University-2024-03-18_19-46-20',
             'lab/Purdue_University-2024-04-10_21-57-29',
             'market/2636_US-52-2024-04-10_23-37-59',
             'market/2801_Northwestern_Ave-2024-04-08_01-18-17',
             'market/2801_Northwestern_Ave-2024-04-08_01-18-171']
type_list = ['Accelerometer', 'Gyroscope', 'Microphone']

def bulit_matrix():
    matrix = []
    for file in file_list:
        for i in range(5):
            tmp_data = []
            for type in type_list:
                data = np.load('Sensor_logger/' + file + '/'
                               + type + str(i) + '.npy')
                tmp_data.extend(data)
            matrix.append(tmp_data)
    Y = ['gym'] * 10 +['home'] * 15 + ['lab'] * 15 + ['market'] * 15
    return matrix, Y

a,b = bulit_matrix()
np.save('processed_X.npy', a)
np.save('processed_Y.npy', b)

a = np.load('processed_X.npy')
b = np.load('processed_Y.npy')
print(b)
