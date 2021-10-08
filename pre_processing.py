##############################PRE-PROSESSING##########################
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from sklearn import preprocessing

from get_data import arff_to_pd_WISDM, txt_to_pd_WISDM

def normalize_data(data):
    data.loc[:,'x-axis'] = data['x-axis'] / data['x-axis'].max()
    data.loc[:,'y-axis'] = data['y-axis'] / data['y-axis'].max()
    data.loc[:,'z-axis'] = data['z-axis'] / data['z-axis'].max()
    data = data.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    return data

def split_train_test_data(data):
#     ## ID 1-28 for training and 28>for testing
    data_test = data[data['user_id'] > 28].copy()
    data_train = data[data['user_id'] <= 28].copy()
    y_test = data_test['activity']
    X_test = data_test.drop(['user_id', 'activity'], axis=1)
    y_train = data_train['activity']
    X_train = data_train.drop(['user_id', 'activity'], axis=1)
    LABEL = 'ActivityEncoded'
    le = preprocessing.LabelEncoder()
    data[LABEL] = le.fit_transform(data['activity'].values.ravel())
    LABELS = list(le.classes_)
    return  X_train, X_test, y_train, y_test, LABELS
    
def pre_prosessing(data, sec=1, overlap_prosent=10, fft = False):
    # Aggregate the data into segments of time_steps size
    # and overlap of overlap_prosent of samples
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
    data = SVM(data)
    # Add frequency domain features
    # create a new dict with features
    feature_data =  {}
    for column_name in  data.drop(['user_id','activity','timestamp'], axis=1):

        mean_list, std_list , max_list, min_list, median_list = ([] for i in range(5))
        abs_sum_list, avg_diff_list , max_diff_list, iqr_list = ([] for i in range(4))
        peak_count_list, skew_list, kurto_list, energy_list = ([] for i in range(4))
        id_list, activity_list, = ([] for i in range(2))

        for i in range(0, len(data)- time_steps, overlap):
            values = data[column_name].values[i: i + time_steps]
            if fft:
                values = np.abs(np.fft.fft(values))[1:time_steps/2+1]

            activity = data['activity'][i: i + time_steps].mode()[0]
            id = data['user_id'][i: i + time_steps].mode()[0]

            (mean, std, max, min, median, abs_sum, 
            avg_diff, max_diff, iqr, peak_count, 
            skew, kurtosis, energy) = create_features(values, column_name)
            mean_list.append(mean)
            std_list.append(std)
            max_list.append(max)
            min_list.append(min)
            median_list.append(median)
            abs_sum_list.append(abs_sum)
            avg_diff_list.append(avg_diff)
            max_diff_list.append(max_diff)
            iqr_list.append(iqr)
            peak_count_list.append(peak_count)
            skew_list.append(skew)
            kurto_list.append(kurtosis)
            energy_list.append(energy)

            activity_list.append(activity)
            id_list.append(id)

        feature_data['user_id'] = id_list
        feature_data['activity'] = activity_list
        feature_data[column_name+'_mean'] = mean_list
        feature_data[column_name+'_std'] = std_list
        feature_data[column_name+'_max'] = max_list
        feature_data[column_name+'_min'] = min_list
        feature_data[column_name+'_median'] = median_list
        feature_data[column_name+'_abs_sum'] = abs_sum_list
        feature_data[column_name+'_avg_diff'] = avg_diff_list
        feature_data[column_name+'_max_diff'] = max_diff_list
        feature_data[column_name+'_iqr'] = iqr_list
        feature_data[column_name+'_peak_count'] = peak_count_list
        feature_data[column_name+'_skew'] = skew_list
        feature_data[column_name+'_kurtosis'] = kurto_list
        feature_data[column_name+'_energy'] = energy_list

    
    feature_df = pd.DataFrame(feature_data)
    print(feature_df)

    return data, feature_df



def create_features(data, column_name=None):
    # Get std mean and more if needed:
    features = []
    mean = np.mean(data)
    std = np.std(data)
    max = np.max(data)
    min = np.min(data)
    median = np.median(data)
    # abs sum
    abs_sum = np.sum(np.abs(data))/len(data)
    # avg absolute diff
    avg_diff =  np.mean(np.absolute(data - np.mean(data)))
    # max-min diff
    max_diff = max-min
    # interquartile range:
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    # number of peaks
    peak_count = len(find_peaks(data)[0])
    # skewness
    skew = stats.skew(data)
    # Kurtosis:
    kurtosis = stats.kurtosis(data)
    # Energy:
    energy = np.sum(data**2)/len(data)


    return (mean, std, max, min, median, abs_sum, 
            avg_diff, max_diff, iqr, peak_count, 
            skew, kurtosis, energy)

def tilt_angle(data):
    Ax = np.arctan2(data['x-axis'],np.sqrt(data['y-axis']**2+data['z-axis']**2))
    Ay = np.arctan2(data['y-axis'],np.sqrt(data['x-axis']**2+data['z-axis']**2))
    df = pd.DataFrame({'Ax': Ax, 'Ay': Ay})
    # print(df)
    data = pd.merge(data, df, left_index=True, right_index=True)
    return data

def normalize_data(data):
    data.loc[:,'x-axis'] = data['x-axis'] / data['x-axis'].max()
    data.loc[:,'y-axis'] = data['y-axis'] / data['y-axis'].max()
    data.loc[:,'z-axis'] = data['z-axis'] / data['z-axis'].max()
    # data = data.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    return data

def SVM(data):
    #SVM = Signal vector magnitude
    data['SVM'] = np.sqrt(data['x-axis']**2+data['y-axis']**2+data['z-axis']**2)
    # print(SMA)
    return data


def create_fft(data):
    x_value = data['x-axis']
    pd.Series(np.fft.fft(x_value[1:len(data)-1])).plot()
    plt.show()
    return data



if __name__ == '__main__':
    file_path = 'Data/WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff'
    data = arff_to_pd_WISDM(file_path)
    # testfolder = 'Data/test/arff_test.csv'
    # # data.to_csv(testfolder)
    # data = txt_to_pd_WISDM()
    # # print(data)
    # data_fft = data
    # data, feature_df = pre_prosessing(data[0:1000])
    # data_fft, fft_feature = pre_prosessing(data_fft[0:1000])
  
    # features = pd.concat([feature_df.reset_index(drop=True),fft_feature.reset_index(drop=True)], axis=1)
    # # data.info()