# LiveData generator for splitting the IMU data in handleable chunks
import csv
import random
import time
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

# data to loop through
plt.style.use('fivethirtyeight')
indx_values = []
x_values = []
y_values = []
z_values = []
q_values = []
counter = 0
index = count()
data = pd.read_csv('Data/30824_OARMH20010.csv')
# data = pd.read_csv('Data/testfile.csv')

def animate(i):
    indx = next(index) #+ 1065000
    indx_values.append(indx)
    counter = next(index)
    print(counter,indx)
    x = data['Accel-X (g)'].iloc[indx]
    y = data[' Accel-Y (g)'].iloc[indx]
    z = data[' Accel-Z (g)'].iloc[indx]
    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

    if counter > 250: # size of window
        # To keep the graph clean and remove old data
        indx_values.pop(0)
        x_values.pop(0)
        y_values.pop(0)
        z_values.pop(0)
        #counter = 0
        plt.cla() # clears the values of the graph
   
    plt.plot(indx_values,x_values, label='Accel-X', color='yellow')
    plt.plot(indx_values,y_values,label='Accel-Y', color='blue')
    plt.plot(indx_values,z_values, label='Accel-X', color='green')
    plt.ylabel('G values')
    plt.xlabel('Samples')
    
    plt.title('Data from IMU')

ani = FuncAnimation(plt.gcf(), animate, interval= 1, frames= 10)
# Lower interval faster speed of animation, in miliseconds
plt.tight_layout()
plt.show()



