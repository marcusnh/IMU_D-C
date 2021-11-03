import numpy as np
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks, butter, lfilter, medfilt, filtfilt
from scipy.signal import freqz
def median_filter(data, f_size):
	lgth, num_signal=data.shape
	f_data=np.zeros([lgth, num_signal])
	for i in range(num_signal):
		f_data[:,i]=medfilt(data[:,i], f_size)
	return f_data

def butter_lowpass_filter(data, cutoff, fs, order,):
    nyg = 0.5 *fs
    normal_cutoff = cutoff / nyg
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def f(x):
    # return np.sin(x) + np.random.normal(scale=0.1, size=len(x))
    return np.sin(2*pi*x*2) + 0.25*np.cos(2*pi*x*35)

if __name__ == '__main__':
    sns.set_style('whitegrid')
    # Number of samples = 600
    N =100
    fs =5000.0
    lowcut = 500.0
    # Meidan filter :
    #spacing:
    T = 1 /fs
    x = np.linspace(0.0, N*T, N)
    y = f(x)
    y_filter = signal.medfilt(y,5)
    # plt.plot(x, y, )
    # plt.plot(x, y_filter)
    # plt.legend(["Unfiltered", "with median filter"], loc ="upper right")
    
    plt.figure(1)
    plt.clf()
    for order in [1, 2, 3, 4, 10]:
        nyg = 0.5 *fs
        normal_cutoff = lowcut / nyg
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
   
    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    # Filter a noisy signal.
    T = 0.05
    nsamples = int( T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_lowpass_filter(x, lowcut, fs, order=6)

    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()