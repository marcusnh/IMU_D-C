import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
# from dask import dataframe as dd



# reading IMU data:
# Assuming sample rate of 25 hz, approx 0.04s between samples
# Test file:
data = pd.read_csv('Data/testfile.csv', delimiter=',', header=0, skipinitialspace = True)
# Data/30824_OARMH20010.csv:
# data = pd.read_csv('Data/30824_OARMH20010.csv', delimiter=',', header=0, sep = r',', skipinitialspace = True)

# Read chunked data:
# print(pd.read_csv('Data/30824_OARMH20010.csv',).shape)
# chunk_list =[]
# STORE = 'store.h5'   # Note: another option is to keep the actual file open
# start = time.time()
# chunk = pd.read_csv('Data/30824_OARMH20010.csv',chunksize=20000)
# end = time.time()
# print("Read csv with chunk: ",(end-start),"sec")
# print(chunk)
# data = pd.concat(chunk)
# print(data.tail())

# Remove day for easier processing:
# for i in range(len(data.index)):
#     # data['Time'][i] = data['Time'][i][12:]
#     data.at[i,'Time'] = int(data.at[i,'Time'])

data.plot()
plt.show()
print(data)
X =data['Accel-X (g)']
Y =data['Accel-Y (g)']
Z =data['Accel-Z (g)']