##################################################################################
#Deep Neural network (DNN) to recognize  the type of movement ( walking, running, 
# sitting, etc) based on a given set of acceleromate data from a mobile device.
# During this script we will use the WISDM dataset.
# Steps:
# 1) Analyse and pre-processing data
# 2) Seperate training and test data set 
# 3) Create model / estimator
# 4) Train model and evaluate
# 5) Evaluate model against test data
###################################################################################

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Standard Parameters:
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)
colors =['brown', 'red', 'green', 'blue', 'yellow', 'orange', ]

## labels we will be using for classificaion:
LABELS ={'Downstairs' , 'Upstairs', 'Jogging', 'Walking', 'Sitting', 'Standing'} 
## Number of steps within one time segment (lenght of time segment):
TIME_PERIODS = 80
## Distance between segments. Determines the amount of overlap between the segments
STEP_DISTANCE = 40

## Data path:
file_path = 'Data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'


# Import data from WISDM:
def get_data(file_path):
    columns = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    #extract data from file_path:
    data = pd.read_csv(file_path, header=None, names=columns)
    # Last column has a ; that needs to be removed and transformed to float
    data['z-axis'].replace(regex=True, inplace=True,to_replace=r';', value=r'')
    data['z-axis'] = data['z-axis'].apply(create_float)
    data.dropna(axis=0, how='any',inplace=True)

    return data
    plot_activity(activity, subset)
def create_float(value):
    try:
        return np.float(value)
    except:
        return np.nan

def show_info(data):
    data['activity'].value_counts().plot(kind='bar', color=colors, title='Number of acitivity samples')
    plt.show()
    data['user_id'].value_counts().plot.bar(rot=0, title='Activity by user')
    plt.show()


data = get_data(file_path)
data.info()
# print(data.head(20))
# show_info(data)
####################################
# 1) Pre-processing the data
####################################
## Need an encoded value for the dataframe
LABEL = 'ActivityEncoded'
le = preprocessing.LabelEncoder()
data[LABEL] = le.fit_transform(data['activity'].values.ravel())
print(data.head())

## The data from WiSDM has a sampling rate of 20 Hz
## use this info to plot the accelerometer data

def show_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(16,12),
            sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')
    fig.suptitle(activity)
    plt.show()



def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y)-np.std(y), max(y)+np.std(y)])
    ax.set_xlim([ min(x), max(x)])
    ax.grid(True)

def plot_activity(data):
    for activity in np.unique(data['activity']):
        subset = data[data['activity'] == activity][:180] 
        # only showing the first 9 seconds (180/20) of the activity
        # of the first ID we find in data
        show_activity(activity,subset)
# plot_activity(data)


#################################################################
#2) split data into training and test sets
# Idea here: let our model learn from a couple of people doing
# all the different activity. We could also have taken some data
# from every person and tested on the remaining data, since some 
# movment is different from person to person, but we consider this
# to be negligible. This has to be considered becuase it will effect
# the performance of the DNN
##################################################################

## ID 1-28 for training and 28>for testing
data_test = data[data['user_id'] > 28].copy()
data_train = data[data['user_id'] <= 28].copy()

## Normalize training data and round numbers
def normalize_data(data):
    data.loc[:,'x-axis'] = data['x-axis'] / data['x-axis'].max()
    data.loc[:,'y-axis'] = data['y-axis'] / data['y-axis'].max()
    data.loc[:,'z-axis'] = data['z-axis'] / data['z-axis'].max()
    data = data.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    return data

data_train = normalize_data(data_train)

# Preparing data for Keras
## Reshape data into segments of size TIME_PERIODS / sampling rate:
## 80/20 = 4 seconds intervals

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
    training_array = np.asarray(training_segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    training_labels = np.asarray(training_labels)

    return training_array, training_labels

x_train, y_train = prepare_data_keras(data_train, TIME_PERIODS, STEP_DISTANCE, LABEL)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)
print(x_train)
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))


# The Keras only excepts a list of values so we have to transform our matrix of
# 80x3 to a list of 240 values
#TODO: remove this if not used
# n_values = num_sensors *num_time_periods
# x_train = x_train.reshape(x_train.shape[0], n_values)
# keras only accept dataytpe float32:
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
#  conduct one-hot-encoding of our labels:
y_train_hot = np_utils.to_categorical(y_train, num_classes)

#################################################################
#3) Create model / estimator
# Here we create the  DNN and have to choose the amount of hidden
# layers and number of nodes. Now that the data is transformed and
# split we can also create a CNN or other complex NN if we want
#################################################################
N_NODES = 100
# sequential keras model withh 100 nodes and 3 layers
model_seq = Sequential()
model_seq.add(Dense(N_NODES, activation='relu'))
model_seq.add(Dense(N_NODES, activation='relu'))
model_seq.add(Dense(N_NODES, activation='relu'))
model_seq.add(Flatten())
model_seq.add(Dense(num_classes, activation='softmax'))


#################################################################
# 4) Train model and evaluate
# using a 80:20 split here between training and validation data
#################################################################
#Training parameters:
BATCH_SIZE = 400
EPOCHS = 50

callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                                    monitor='val_loss', save_best_only=True), 
    keras.callbacks.TensorBoard(log_dir='./logs'),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1), #early_stopping_monitor
]
# config the model  
model_seq.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])
# fit the model:
history = model_seq.fit(x_train, y_train_hot, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks=callbacks_list, validation_split= 0.2, verbose=1)
print(model_seq.summary())

