##############################PRE-PROSESSING##########################
 
import numpy as np
import pandas as pd
import scipy as sp
import random
import matplotlib.pyplot as plt
import scipy.fftpack
from math import ceil
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks, butter, lfilter, medfilt, filtfilt

from sklearn import preprocessing


from get_data import arff_to_pd_WISDM, txt_to_pd_WISDM
# from visualize_data import activity_data_per_user, compare_user_activitys, show_activity
# from visualize_data import activity_difference_between_users, total_activities

def normalize_data_old(data):
    data.loc[:,'x-axis'] = data['x-axis'] / data['x-axis'].max()
    data.loc[:,'y-axis'] = data['y-axis'] / data['y-axis'].max()
    data.loc[:,'z-axis'] = data['z-axis'] / data['z-axis'].max()
    # data = data.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    return data

def normalize_data(data):
    drop_columns =['timestamp', 'user_id', 'activity']
    for column in data.drop(columns=drop_columns):
        numerator = (data[column]-data[column].min())
        denominator = (data[column].max()-data[column].min())
        data[column] = 2*(numerator/denominator)-1
        
    return data
    
def activity_encoded(data):
     ## Need an encoded value for the dataframe
    LABEL = 'ActivityEncoded'
    le = preprocessing.LabelEncoder()
    data[LABEL] = le.fit_transform(data['activity'].values.ravel())
    LABELS = list(le.classes_)
    return data, LABELS, LABEL

def split_train_test_data(data,ratio):
    # split the data per subject, but randomize order
    # test and validation will not be complete equal
    # depends on size of user_id data and ratio
#     ## ID 1-28 for training and 28>for testing
    # data_test = data[data['user_id'] > 28].copy()
    # data_train = data[data['user_id'] <= 28].copy()
    user_list =data.user_id.unique()
    # user_list = random.shuffle(user_list.tolist())
    user_list = user_list.tolist()
    random.shuffle(user_list)
    train_size = int(len(user_list)*ratio)
    test_size = train_size + int(ceil(len(user_list)*(1-ratio)/2)) 

    train_list = user_list[:train_size]
    test_list = user_list[train_size:test_size]
    val_list = user_list[test_size:]
    print(len(train_list))
    print(len(test_list))
    print(len(val_list))
    data_train = data[data['user_id'].isin(train_list)]
    data_test = data[data['user_id'].isin(test_list)]
    data_val = data[data['user_id'].isin(val_list)]
    
    y_test = data_test['activity']
    y_train = data_train['activity']
    y_val = data_val['activity']

    X_test = data_test.drop(['user_id', 'activity'], axis=1)
    X_train = data_train.drop(['user_id', 'activity'], axis=1)
    X_val = data_val.drop(['user_id', 'activity'], axis=1)
    
    LABEL = 'ActivityEncoded'
    le = preprocessing.LabelEncoder()
    data[LABEL] = le.fit_transform(data['activity'].values.ravel())
    LABELS = list(le.classes_)
    return  X_train, X_test, X_val, y_train, y_test, y_val, LABELS

def extract_windows(data, sec=1, overlap_prosent=10):
    # Aggregate the data into segments of time_steps size
    # and overlap of overlap_prosent of samples
    # Extracty the wanted features from segment
    time_steps = int(sec*20)
    overlap = time_steps - int(time_steps*(overlap_prosent/100))
    if overlap_prosent>100 or overlap_prosent<0:
        print("Invalid Input Entered")
        print("Overlap_prosent value must be between [0,100]")
        print("And the total overlap can't be zero, must at least be one")
        exit(0) 
    if overlap <=0:
        print('Overlap is sat to 1 sample since it was set to 0 which is not possible')
        print('One is the smallest possible overlap')
        overlap =1
    data, LABELS, LABEL = activity_encoded(data)
 
    segments = []
    labels = []

    drop_columns =['timestamp', 'user_id', 'activity', LABEL]
    N_FEATURES = len(data.drop(columns=drop_columns).columns)


    for i in range(0, len(data)- time_steps, overlap):

        label = stats.mode(data[LABEL][i: i + time_steps])[0][0]
        labels.append(label)
   
        x_values = data['x-axis'].values[i: i + time_steps]
        y_values = data['y-axis'].values[i: i + time_steps]
        z_values = data['z-axis'].values[i: i + time_steps]
      
        # ax_values = data['Ax'].values[i: i + time_steps]
        # ay_values = data['Ay'].values[i: i + time_steps]
        # SVM_values = data['SVM'].values[i: i + time_steps]

        segments.append([x_values, y_values, z_values, ])
        #segments.append([x_values, y_values, z_values, #])
        #                          ax_values, ay_values, SVM_values])
  

    # reshape into an array with x rows and columns equal to time_steps, and seperate for each feature 
    segments_reshaped = np.asarray(segments, 
                        dtype=np.float32).reshape(-1, time_steps, 
                        N_FEATURES)
    labels = np.asarray(labels)

    return segments_reshaped, labels, LABELS


def prepare_data_keras(data, time_steps, step, labels):
    # number of features: x, y and z
    N_FEATURES = 3
    training_segments = []
    training_labels = []
    for i in range(0, len(data)- time_steps, step):
        x_values = data['x-axis'].values[i: i + time_steps]
        y_values = data['y-axis'].values[i: i + time_steps]
        z_values = data['z-axis'].values[i: i + time_steps]
        # the label can be different throughout the segment so we
        # choose the most used label in the segment
        label = stats.mode(data[labels][i: i + time_steps])[0][0]
        
        training_segments.append([x_values, y_values, z_values])
        training_labels.append(label)

    # reshape into an array with x rows and columns equal to time_steps, and seperate for each feature 
    training_array = np.asarray(training_segments, 
                                dtype=np.float32).reshape(-1, time_steps, 
                                N_FEATURES)
    training_labels = np.asarray(training_labels)

    return training_array, training_labels



def tilt_angle(data):
    Ax = np.arctan2(data['x-axis'],np.sqrt(data['y-axis']**2+data['z-axis']**2))
    Ay = np.arctan2(data['y-axis'],np.sqrt(data['x-axis']**2+data['z-axis']**2))
    df = pd.DataFrame({'Ax': Ax, 'Ay': Ay})
    # print(df)
    data = pd.merge(data, df, left_index=True, right_index=True)
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

################################################################################
# FILTERING:
################################################################################
def butter_lowpass_filter(data, cutoff, fs, order,):
    nyq = 0.5 *fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 *fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def median_filter(data, f_size):
	lgth, num_signal=data.shape
	f_data=np.zeros([lgth, num_signal])
	for i in range(num_signal):
		f_data[:,i]=medfilt(data[:,i],f_size)
	return f_data

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def filter_butter(data, cutoff=0.4, type ='low'):
   #sampling frequency:
    fs = 20
    # cutoff frequency: 20/50 = 40%
    #Cutoff share
    cutoff = int(fs *cutoff) 

    tot_samples = len(data)
    tot_periode = tot_samples/fs
    order = 3
    data_filtered = data.copy()

    drop_columns =['timestamp', 'user_id', 'activity']
    # filtering:
    for column in data_filtered.drop(columns=drop_columns):

        if type == 'low':
            data_filtered[column] = butter_lowpass_filter(data_filtered[column], 
                                                cutoff=cutoff, fs=fs,
                                                order =order)
        elif type =='high':
            data_filtered[column] = butter_highpass_filter(data_filtered[column], 
                                                cutoff=cutoff, fs=fs,
                                                order =order)
        else:
            print('Wrong input type')
            exit(0)

    return data_filtered

def filter_data(data, fs_share=0.4, nr_medfil=3):
    #sampling frequency:
    fs = 20
    # cutoff frequency: 20/50 = 40%
    #All frequencies above cutoff are filtered out.
    cutoff = int(fs *fs_share) 
    nyq = 0.5 *fs
    tot_samples = len(data)
    tot_periode = tot_samples/fs

    # order of filter:
    order = 3
    # data = data.drop(columns=['timestamp', 'user_id', 'activity'])
    data_new = data.copy()

    drop_columns =['timestamp', 'user_id', 'activity']
    # filtering:
    for column in data_new.drop(columns=drop_columns):

        data_new[column] = medfilt(data_new[column], nr_medfil )
        data_new[column] = butter_lowpass_filter(data_new[column], 
                                                cutoff=cutoff, fs=fs,
                                                order =order)

    # Plot data before and after
    nr_df_columns = len(data.drop(columns=drop_columns).columns)
    fig, axes = plt.subplots(nrows=1, ncols=nr_df_columns, figsize=(10, 5))
    plt.suptitle('Comparing filter and unfiltered data:')
    colors = plt.rcParams["axes.prop_cycle"]()
    c = next(colors)['color']
    counter = 0
    for j, k in zip(data.drop(columns=drop_columns), data_new.drop(columns=drop_columns)):
        title = j + ' vs. ' + k

        if len(data.drop(columns=drop_columns).columns) > 1:
            data[j].plot(ax=axes[counter], label=j, )
            data_new[k].plot(title=title, ax=axes[counter], label=k, )
            axes[counter].legend(loc="upper right")
        else:
            data[j].plot(ax=axes, label=j, )
            data_new[k].plot(title=title, ax=axes, label=k, )

        # c = next(colors)['color']
       
        counter +=1


    # plt.show()
        
    return data_new

def frequency_domain(data, fs):
    # Frequency Domain Analysis
    nyq = 0.5 * fs  # Nyquist Frequenc
    N = len (data)
    fstep = fs/N
    f = np.linspace(0, (N-1)*fstep, N)

    print(data)
    print(type(data['x-axis'].values))
    print()
    x = data['x-axis'].values
    print(x)
    fft = scipy.fftpack.fft(x)
    # Power spectral density:
    x_psd = np.abs(x) ** 2
    fft_freq =  sp.fftpack.fftfreq(len(x_psd), 1. /fs)
    i = fft_freq > 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.plot(fft_freq, x_psd)
    # # ax.plot(fft_freq[i], 10 * np.log10(x_psd[i]))
    # ax.set_xlim(0, 30)
    # ax.set_xlabel('Frequency hz')
    # ax.set_ylabel('PSD (dB)')
    # plt.show()

    fft_mag = np.abs(fft)
    f_plot = f[0:int(N/2 +1)]
    fft_mag_plot = 2 * fft_mag[0:int(N/2 +1)]
    fft_mag_plot[0] = fft_mag_plot[0]/2 # DC component must be reduced
    # plt.plot(fft)
    plt.plot(f_plot, fft_mag_plot)
    plt.xlabel("freq(Hz)")
    plt.grid()
    # plt.show()



if __name__ == '__main__':
    data = txt_to_pd_WISDM()
    # data = SVM(data)
    # activity_data_per_user(data)
    frequency_domain(data, fs=20)
    # # total_activities(data)
    user = [33, 1]
    
    # # # user = data['user_id'].unique()
    # print(user)
    activity = 'Walking'
    # # Nomralize data set, not sure it really matters
    data = normalize_data(data)
    user_data =data[data['user_id']==user[0]]
    activity_data = user_data[user_data['activity'] == activity]
    # # print(data_norm)
    # # show_activity(data, activity, user[0], samples=128)
    # # show_activity(data_norm, activity, user[0], samples=128)
    

    # # activity_difference_between_users(data,user[10:15],activity,samples=200)
    # # compare_user_activitys(data_norm, user[0], ) #activity=activity)
    
    # # # show_activity(data, activity, user, samples=128)
    data_filtered = filter_data(activity_data,fs_share =0.4, nr_medfil=21)
    # print(data_filtered)
    frequency_domain(data_filtered, fs=20)
    plt.show()
    # # activity_difference_between_users(data_filtered,user[10:15],activity,samples=200)

    # # # show_activity(data_filtered, activity, user, samples=128)
    # # plt.show()

    # # testfolder = 'Data/test/arff_test.csv'
    # # # data.to_csv(testfolder)
    # # data = txt_to_pd_WISDM()
    # # # print(data)
    # # data_fft = data
    # # data, feature_df = pre_prosessing(data[0:1000])
    # # data_fft, fft_feature = pre_prosessing(data_fft[0:1000])
  
    # # features = pd.concat([feature_df.reset_index(drop=True),fft_feature.reset_index(drop=True)], axis=1)
    # # # data.info()