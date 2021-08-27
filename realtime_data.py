
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as ani
from itertools import count

# data = pd.read_csv('Data/testfile.csv', index_col = 'Time', delimiter=',', header=0, sep = r',', skipinitialspace = True)
# fig =plt.figure()
# color = ['red', 'green', 'blue']
# plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #rotate the x-axis values
# plt.subplots_adjust(bottom = 0.2, top = 0.9) #ensuring the dates (on the x-axis) fit in the screen
# plt.ylabel('G - power')
# plt.xlabel('sample')
plt.style.use('fivethirtyeight')

# def buildmebarchart(i=int):
#     plt.legend(data.columns[0:])
#     p = plt.plot(data[:i].index, data[:i].values) #note it only returns the dataset, up to the point i
  
#     for i in range(0,3):
#         p[i].set_color(color[i]) #set the colour of each curve

def animate(i):
    data = pd.read_csv('Data/run_file.csv', delimiter=',', header=0, sep = r',', skipinitialspace = True)
    print(data)
    x = data['Time']
    y1 = data['Accel-X (g)']
    y2 = data['Accel-Y (g)']
    y3 = data['Accel-Z (g)']
    # clear current axsis
    plt.cla()
    plt.plot(x, y1, label='Accel-X')
    plt.plot(x, y2, label='Accel-Y')
    plt.plot(x, y3, label='Accel-Z')
    plt.tight_layout()


    
        
# animator = ani.FuncAnimation(fig, buildmebarchart, interval = 0.000000001)
animator = ani.FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()

plt.show()