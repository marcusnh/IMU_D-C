import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings

from get_data import txt_to_pd_WISDM, arff_to_pd_WISDM
from visualize_data import execute_TSNE

def tilt_angle(data):
    Ax = np.arctan2(data['x-axis'],np.sqrt(data['y-axis']**2+data['z-axis']**2))
    Ay = np.arctan2(data['y-axis'],np.sqrt(data['x-axis']**2+data['z-axis']**2))
    df = pd.DataFrame({'Ax': Ax, 'Ay': Ay})
    # print(df)
    data = pd.merge(data, df, left_index=True, right_index=True)
    return data

def feature_extraction(data, sec=10, overlap_prosent=50):
    x_values = []
    y_values = []
    z_values = []
    ax_values = []
    ay_values = []
    activity_labels = []
    id_values = []


      # Extracty the wanted features from segment
    time_steps = sec*20
    overlap =int(time_steps*(1-overlap_prosent/100))
    if overlap_prosent>100 or overlap_prosent<0:
        print("Invalid Input Entered")
        print("Overlap_prosent value must be between [0,100]")
        print("And the total overlap can't be zero, must at least be one")
        exit(0) 
    # 20Hz (1 sample every 50ms) for WISDM
    if overlap <=0:
        print('Overlap is sat to 1 sample since it was set to 0 which is not possible')
        print('One is the smallest possible overlap')
        overlap =1
    
    data = (tilt_angle(data))
    # windowing function:
    for i in range(0, len(data)- time_steps, overlap):
        xs = data['x-axis'].values[i: i + time_steps]
        ys = data['y-axis'].values[i: i + time_steps]
        zs = data['z-axis'].values[i: i + time_steps]
        ax = data['Ax']. values[i: i + time_steps]
        ay = data['Ax']. values[i: i + time_steps]
        activity = data['activity'][i: i + time_steps].mode()[0]
        id = data['user_id'][i: i + time_steps].mode()[0]

        x_values.append(xs)
        y_values.append(ys)
        z_values.append(zs)
        ax_values.append(ax)
        ay_values.append(ay)
        activity_labels.append(activity)
        id_values.append(id)

    # Statistical Features on raw x, y and z in time domain
    feature_df = pd.DataFrame()
    feature_df['user_id'] = id_values
    feature_df['activity'] = activity_labels
    #fast fourier transform:
    # converting the signals from time domain to frequency domain using FFT
    x_values_fft = pd.Series(x_values).apply(lambda x: np.abs(np.fft.fft(x))[1:int(time_steps/2+1)])
    y_values_fft = pd.Series(y_values).apply(lambda x: np.abs(np.fft.fft(x))[1:int(time_steps/2+1)])
    z_values_fft = pd.Series(z_values).apply(lambda x: np.abs(np.fft.fft(x))[1:int(time_steps/2+1)])

    # mean
    feature_df['x_mean'] = pd.Series(x_values).apply(lambda x: x.mean())
    feature_df['y_mean'] = pd.Series(y_values).apply(lambda x: x.mean())
    feature_df['z_mean'] = pd.Series(z_values).apply(lambda x: x.mean())
    feature_df['ax_mean'] = pd.Series(ax_values).apply(lambda x: x.mean())
    feature_df['ay_mean'] = pd.Series(ay_values).apply(lambda x: x.mean())
    # FFT mean
    feature_df['x_mean_fft'] = pd.Series(x_values_fft).apply(lambda x: x.mean())
    feature_df['y_mean_fft'] = pd.Series(y_values_fft).apply(lambda x: x.mean())
    feature_df['z_mean_fft'] = pd.Series(z_values_fft).apply(lambda x: x.mean())
    # std dev
    feature_df['x_std'] = pd.Series(x_values).apply(lambda x: x.std())
    feature_df['y_std'] = pd.Series(y_values).apply(lambda x: x.std())
    feature_df['z_std'] = pd.Series(z_values).apply(lambda x: x.std())
    # FFT std dev
    feature_df['x_std_fft'] = pd.Series(x_values_fft).apply(lambda x: x.std())
    feature_df['y_std_fft'] = pd.Series(y_values_fft).apply(lambda x: x.std())
    feature_df['z_std_fft'] = pd.Series(z_values_fft).apply(lambda x: x.std())
    
    # avg absolute diff
    feature_df['x_aad'] = pd.Series(x_values).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    feature_df['y_aad'] = pd.Series(y_values).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    feature_df['z_aad'] = pd.Series(z_values).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    # FFT avg absolute diff
    feature_df['x_aad_fft'] = pd.Series(x_values_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    feature_df['y_aad_fft'] = pd.Series(y_values_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    feature_df['z_aad_fft'] = pd.Series(z_values_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    feature_df['x_min'] = pd.Series(x_values).apply(lambda x: x.min())
    feature_df['y_min'] = pd.Series(y_values).apply(lambda x: x.min())
    feature_df['z_min'] = pd.Series(z_values).apply(lambda x: x.min())
    # FFT min
    feature_df['x_min_fft'] = pd.Series(x_values_fft).apply(lambda x: x.min())
    feature_df['y_min_fft'] = pd.Series(y_values_fft).apply(lambda x: x.min())
    feature_df['z_min_fft'] = pd.Series(z_values_fft).apply(lambda x: x.min())

    # max
    feature_df['x_max'] = pd.Series(x_values).apply(lambda x: x.max())
    feature_df['y_max'] = pd.Series(y_values).apply(lambda x: x.max())
    feature_df['z_max'] = pd.Series(z_values).apply(lambda x: x.max())
    # FFT max
    feature_df['x_max_fft'] = pd.Series(x_values_fft).apply(lambda x: x.max())
    feature_df['y_max_fft'] = pd.Series(y_values_fft).apply(lambda x: x.max())
    feature_df['z_max_fft'] = pd.Series(z_values_fft).apply(lambda x: x.max())

    # max-min diff
    feature_df['x_maxmin_diff'] = feature_df['x_max'] - feature_df['x_min']
    feature_df['y_maxmin_diff'] = feature_df['y_max'] - feature_df['y_min']
    feature_df['z_maxmin_diff'] = feature_df['z_max'] - feature_df['z_min']
    # FFT max-min diff
    feature_df['x_maxmin_diff_fft'] = feature_df['x_max_fft'] - feature_df['x_min_fft']
    feature_df['y_maxmin_diff_fft'] = feature_df['y_max_fft'] - feature_df['y_min_fft']
    feature_df['z_maxmin_diff_fft'] = feature_df['z_max_fft'] - feature_df['z_min_fft']

    # median
    feature_df['x_median'] = pd.Series(x_values).apply(lambda x: np.median(x))
    feature_df['y_median'] = pd.Series(y_values).apply(lambda x: np.median(x))
    feature_df['z_median'] = pd.Series(z_values).apply(lambda x: np.median(x))
    # FFT median
    feature_df['x_median_fft'] = pd.Series(x_values_fft).apply(lambda x: np.median(x))
    feature_df['y_median_fft'] = pd.Series(y_values_fft).apply(lambda x: np.median(x))
    feature_df['z_median_fft'] = pd.Series(z_values_fft).apply(lambda x: np.median(x))

    # median abs dev 
    feature_df['x_mad'] = pd.Series(x_values).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    feature_df['y_mad'] = pd.Series(y_values).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    feature_df['z_mad'] = pd.Series(z_values).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    # FFT median abs dev 
    feature_df['x_mad_fft'] = pd.Series(x_values_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    feature_df['y_mad_fft'] = pd.Series(y_values_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    feature_df['z_mad_fft'] = pd.Series(z_values_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    feature_df['x_IQR'] = pd.Series(x_values).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    feature_df['y_IQR'] = pd.Series(y_values).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    feature_df['z_IQR'] = pd.Series(z_values).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    # FFT Interquartile range
    feature_df['x_IQR_fft'] = pd.Series(x_values_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    feature_df['y_IQR_fft'] = pd.Series(y_values_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    feature_df['z_IQR_fft'] = pd.Series(z_values_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
   
    # negtive count
    feature_df['x_neg_count'] = pd.Series(x_values).apply(lambda x: np.sum(x < 0))
    feature_df['y_neg_count'] = pd.Series(y_values).apply(lambda x: np.sum(x < 0))
    feature_df['z_neg_count'] = pd.Series(z_values).apply(lambda x: np.sum(x < 0))

    # positive count
    feature_df['x_pos_count'] = pd.Series(x_values).apply(lambda x: np.sum(x > 0))
    feature_df['y_pos_count'] = pd.Series(y_values).apply(lambda x: np.sum(x > 0))
    feature_df['z_pos_count'] = pd.Series(z_values).apply(lambda x: np.sum(x > 0))

    # values above mean
    feature_df['x_above_mean'] = pd.Series(x_values).apply(lambda x: np.sum(x > x.mean()))
    feature_df['y_above_mean'] = pd.Series(y_values).apply(lambda x: np.sum(x > x.mean()))
    feature_df['z_above_mean'] = pd.Series(z_values).apply(lambda x: np.sum(x > x.mean()))
     # FFT values above mean
    feature_df['x_above_mean_fft'] = pd.Series(x_values_fft).apply(lambda x: np.sum(x > x.mean()))
    feature_df['y_above_mean_fft'] = pd.Series(y_values_fft).apply(lambda x: np.sum(x > x.mean()))
    feature_df['z_above_mean_fft'] = pd.Series(z_values_fft).apply(lambda x: np.sum(x > x.mean()))

    # number of peaks
    feature_df['x_peak_count'] = pd.Series(x_values).apply(lambda x: len(find_peaks(x)[0]))
    feature_df['y_peak_count'] = pd.Series(y_values).apply(lambda x: len(find_peaks(x)[0]))
    feature_df['z_peak_count'] = pd.Series(z_values).apply(lambda x: len(find_peaks(x)[0]))
    # FFT number of peaks
    feature_df['x_peak_count_fft'] = pd.Series(x_values_fft).apply(lambda x: len(find_peaks(x)[0]))
    feature_df['y_peak_count_fft'] = pd.Series(y_values_fft).apply(lambda x: len(find_peaks(x)[0]))
    feature_df['z_peak_count_fft'] = pd.Series(z_values_fft).apply(lambda x: len(find_peaks(x)[0]))

    # skewness
    feature_df['x_skewness'] = pd.Series(x_values).apply(lambda x: stats.skew(x))
    feature_df['y_skewness'] = pd.Series(y_values).apply(lambda x: stats.skew(x))
    feature_df['z_skewness'] = pd.Series(z_values).apply(lambda x: stats.skew(x))
    # FFT skewness
    feature_df['x_skewness_fft'] = pd.Series(x_values_fft).apply(lambda x: stats.skew(x))
    feature_df['y_skewness_fft'] = pd.Series(y_values_fft).apply(lambda x: stats.skew(x))
    feature_df['z_skewness_fft'] = pd.Series(z_values_fft).apply(lambda x: stats.skew(x))

    # kurtosis
    feature_df['x_kurtosis'] = pd.Series(x_values).apply(lambda x: stats.kurtosis(x))
    feature_df['y_kurtosis'] = pd.Series(y_values).apply(lambda x: stats.kurtosis(x))
    feature_df['z_kurtosis'] = pd.Series(z_values).apply(lambda x: stats.kurtosis(x))
    # FFT kurtosis
    feature_df['x_kurtosis_fft'] = pd.Series(x_values_fft).apply(lambda x: stats.kurtosis(x))
    feature_df['y_kurtosis_fft'] = pd.Series(y_values_fft).apply(lambda x: stats.kurtosis(x))
    feature_df['z_kurtosis_fft'] = pd.Series(z_values_fft).apply(lambda x: stats.kurtosis(x))

    # energy
    feature_df['x_energy'] = pd.Series(x_values).apply(lambda x: np.sum(x**2)/100)
    feature_df['y_energy'] = pd.Series(y_values).apply(lambda x: np.sum(x**2)/100)
    feature_df['z_energy'] = pd.Series(z_values).apply(lambda x: np.sum(x**2/100))
    # FFT energy
    feature_df['x_energy_fft'] = pd.Series(x_values_fft).apply(lambda x: np.sum(x**2)/50)
    feature_df['y_energy_fft'] = pd.Series(y_values_fft).apply(lambda x: np.sum(x**2)/50)
    feature_df['z_energy_fft'] = pd.Series(z_values_fft).apply(lambda x: np.sum(x**2/50))
    # avg resultant
    feature_df['avg_result_accl'] = [i.mean() for i in ((pd.Series(x_values)**2 + pd.Series(y_values)**2 + pd.Series(z_values)**2)**0.5)]
    # FFT avg resultant
    feature_df['avg_result_accl_fft'] = [i.mean() for i in ((pd.Series(x_values_fft)**2 + pd.Series(y_values_fft)**2 + pd.Series(z_values_fft)**2)**0.5)]

    # signal magnitude area
    feature_df['sma'] =    pd.Series(x_values).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_values).apply(lambda x: np.sum(abs(x)/100)) \
                    + pd.Series(z_values).apply(lambda x: np.sum(abs(x)/100))
    
    # FFT Signal magnitude area
    feature_df['sma_fft'] = pd.Series(x_values_fft).apply(lambda x: np.sum(abs(x)/50)) + pd.Series(y_values_fft).apply(lambda x: np.sum(abs(x)/50)) \
                        + pd.Series(z_values_fft).apply(lambda x: np.sum(abs(x)/50))
    #Capturing indices
    # Max Indices and Min indices 

    # index of max value in time domain
    feature_df['x_argmax'] = pd.Series(x_values).apply(lambda x: np.argmax(x))
    feature_df['y_argmax'] = pd.Series(y_values).apply(lambda x: np.argmax(x))
    feature_df['z_argmax'] = pd.Series(z_values).apply(lambda x: np.argmax(x))

    # index of min value in time domain
    feature_df['x_argmin'] = pd.Series(x_values).apply(lambda x: np.argmin(x))
    feature_df['y_argmin'] = pd.Series(y_values).apply(lambda x: np.argmin(x))
    feature_df['z_argmin'] = pd.Series(z_values).apply(lambda x: np.argmin(x))

    # absolute difference between above indices
    feature_df['x_arg_diff'] = abs(feature_df['x_argmax'] - feature_df['x_argmin'])
    feature_df['y_arg_diff'] = abs(feature_df['y_argmax'] - feature_df['y_argmin'])
    feature_df['z_arg_diff'] = abs(feature_df['z_argmax'] - feature_df['z_argmin'])

    # index of max value in frequency domain
    feature_df['x_argmax_fft'] = pd.Series(x_values_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))
    feature_df['y_argmax_fft'] = pd.Series(y_values_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))
    feature_df['z_argmax_fft'] = pd.Series(z_values_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))

    # index of min value in frequency domain
    feature_df['x_argmin_fft'] = pd.Series(x_values_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))
    feature_df['y_argmin_fft'] = pd.Series(y_values_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))
    feature_df['z_argmin_fft'] = pd.Series(z_values_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))

    # absolute difference between above indices
    feature_df['x_arg_diff_fft'] = abs(feature_df['x_argmax_fft'] - feature_df['x_argmin_fft'])
    feature_df['y_arg_diff_fft'] = abs(feature_df['y_argmax_fft'] - feature_df['y_argmin_fft'])
    feature_df['z_arg_diff_fft'] = abs(feature_df['z_argmax_fft'] - feature_df['z_argmin_fft'])

    return feature_df
if __name__ == '__main__':
    data = txt_to_pd_WISDM()
    feature_df = feature_extraction(data, sec=10, overlap_prosent=0)
    print(feature_df)
    test_file_path = 'Data/test/WISDM_feature.csv'
    feature_df.to_csv(test_file_path)

    # data_arff = arff_to_pd_WISDM()
    # print(data_arff)
    # execute_TSNE(data_arff)

