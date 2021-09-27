import pandas as pd
import numpy as np

import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

from HAR_ML import get_data

def tilt_angle(data):
    Ax = np.arctan2(data['x-axis'],np.sqrt(data['y-axis']**2+data['z-axis']**2))
    Ay = np.arctan2(data['y-axis'],np.sqrt(data['x-axis']**2+data['z-axis']**2))
    print(Ax)
    print(type(Ax))
    print(type(data))
    df = pd.DataFrame({'Ax': Ax, 'Ay': Ay})
    print(df)
    data = pd.merge(data, df, left_index=True, right_index=True)
    print(data)
    return data

def SMA(data):
    data['magnitude'] = np.sqrt(data['x-axis']**2+data['y-axis']**2+data['z-axis']**2)
    SMA = data['magnitude'].sum()/len(data)
    print(SMA)
    return data, SMA






if __name__ == '__main__':
    file_path = 'Data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    data = get_data(file_path)
    print(data)
    # tilt_angle(data)
    SMA(data)
    ## ID 1-28 for training and 28>for testing
    data_test = data[data['user_id'] > 28].copy()
    data_train = data[data['user_id'] <= 28].copy()

