import pandas as pd
import numpy as np

def read_csv(file_path, file_name):
    # Read the specified columns from the CSV file
    data = pd.read_csv(file_path + file_name +'.csv', usecols=['z', 'y', 'x'], nrows=60000)

    # Initialize dictionaries to store the x, y, z data for each of the five groups
    group_data = {i: {'x': [], 'y': [], 'z': []} for i in range(10)}

    # Populate the dictionaries with data from the specified columns
    for index, row in data.iterrows():
        group = index % 10
        group_data[group]['x'].append(row['x'])
        group_data[group]['y'].append(row['y'])
        group_data[group]['z'].append(row['z'])
    for i in range(10):
        results_list = average_std(group_data[i])
        np.save(file_path + file_name + str(i) + '.npy', results_list)

def read_microphone(file_path, file_name):
    # Read the specified columns from the CSV file
    data = pd.read_csv(file_path + file_name +'.csv', usecols=['dBFS'], nrows=60000)

    # Initialize dictionaries to store the x, y, z data for each of the five groups
    group_data = {i: {'dBFS': []} for i in range(10)}

    # Populate the dictionaries with data from the specified columns
    for index, row in data.iterrows():
        group = index % 10
        group_data[group]['dBFS'].append(row['dBFS'])
    for i in range(10):
        results_list = average_micro(group_data[i])
        np.save(file_path + file_name + str(i) + '.npy', results_list)

def read_Heart(file_path, file_name):
    # Read the specified columns from the CSV file
    data = pd.read_csv(file_path + file_name +'.csv', usecols=['bpm'])
    results_list = average_bpm(data)
    for i in range(10):
        np.save(file_path + file_name + str(i) + '.npy', results_list)

def average_bpm(group_data):
    '''
    get the average of bpm
    :param group_data: one group of data
    :return:
    '''
    x_list = group_data['bpm']
    x_average = np.mean(x_list)
    x_std = np.std(x_list)
    results_list = [x_average, x_std]
    return results_list

def average_micro(group_data):
    '''
    get the average of micro dBFS
    :param group_data: one group of data
    :return:
    '''
    x_list = group_data['dBFS']
    x_average = np.mean(x_list)
    x_std = np.std(x_list)
    results_list = [x_average, x_std]
    return results_list

def average_std(group_data):
    '''
    get the average of each direction
    :param group_data: one group of data
    :return:
    '''
    x_list = group_data['x']
    y_list = group_data['y']
    z_list = group_data['z']
    x_average = np.mean(x_list)
    x_std = np.std(x_list)
    y_average = np.mean(y_list)
    y_std = np.std(y_list)
    z_average = np.mean(z_list)
    z_std = np.std(z_list)
    results_list = [x_average, x_std, y_average, y_std, z_average, z_std]
    return results_list






file_path = 'Sensor_logger/lab/Purdue_University-2024-04-10_21-57-29/'
file_name = 'Accelerometer'
read_csv(file_path, file_name)
file_name = 'Gyroscope'
read_csv(file_path, file_name)
file_name = 'Microphone'
read_microphone(file_path, file_name)
file_name = 'HeartRate'
read_Heart(file_path, file_name)