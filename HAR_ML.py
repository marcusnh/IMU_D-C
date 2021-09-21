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
import random
import pickle
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Embedding, LSTM
from keras.layers import Conv2D, MaxPooling2D, Bidirectional
from keras.utils import np_utils

import tensorflow as tf

# Standard Parameters:
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
# # color template:
colors =['brown', 'red', 'green', 'blue', 'yellow', 'orange', ]
N_FEATURES = 3
## labels we will be using for classificaion:
LABELS ={'Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'} 
## Number of steps within one time segment (lenght of time segment):
TIME_PERIODS = 80
## Distance between segments. Determines the amount of overlap between the segments
STEP_DISTANCE = 40

## Data path to WISDM:
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

def normalize_data(data):
    data.loc[:,'x-axis'] = data['x-axis'] / data['x-axis'].max()
    data.loc[:,'y-axis'] = data['y-axis'] / data['y-axis'].max()
    data.loc[:,'z-axis'] = data['z-axis'] / data['z-axis'].max()
    data = data.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    return data



def prepare_data_keras(data, time_steps, step, labels):
    # number of features: x, y and z
    
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

def create_confusion_matrix(vali, predict, LABELS):
    matrix = metrics.confusion_matrix(vali, predict) 
    plt.figure( figsize=(6, 4))
    sns.heatmap(matrix, cmap='coolwarm', linecolor='white',linewidths=1, 
                xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def create_DNN_model(N_NODES, N_LAYERS, COSTUME, N_CLASSES):
    model_seq = Sequential()
    if (COSTUME):
        model_seq.add(Dense(N_NODES*2, activation='relu'))
        model_seq.add(Dense(N_NODES, activation='relu'))
        model_seq.add(Dense(N_NODES/2, activation='relu'))
        model_seq.add(Dense(N_NODES/4, activation='relu'))
        model_seq.add(Flatten())
        model_seq.add(Dense(N_CLASSES, activation='softmax'))
    else:
        NODES = random.randint(N_NODES/2, N_NODES*2)
        for i in range(N_LAYERS):
            model_seq.add(Dense(N_NODES, activation='relu'))
            
        model_seq.add(Flatten())
        model_seq.add(Dense(N_CLASSES, activation='softmax'))

    return model_seq

def create_LSTM_model(N_NODES, N_CLASSES):
    #RNN
    model = Sequential()
    model.add(Dense(N_NODES, activation='relu'))
    model.add(Dense(N_NODES, activation='relu'))
    model.add(LSTM(N_NODES, activation='relu',return_sequences=True,
                        input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(LSTM(N_NODES, activation='relu'))
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model

def create_bidir_LSTM_model(N_NODES, N_CLASSES):
    #BRNN
    model = Sequential()
    model.add(Dense(N_NODES, activation='relu'))
    model.add(Dense(N_NODES, activation='relu'))
    model.add(Dropout(0.2,))
    model.add(Bidirectional(LSTM(round(N_NODES/5), activation='relu',
                                return_sequences=True),
                                input_shape=(TIME_PERIODS,N_FEATURES)))

    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))


    return model


def train_model(BATCH_SIZE, EPOCHS, file_name, file_type, model, x_train, y_train):
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath='best_'+file_name+'.{epoch:02d}-{val_accuracy:.2f}.h5',
                                        monitor='accuracy', save_best_only=True), 
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1, patience=10, 
                                      min_delta=0.01),]

     # config the model  
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])
    # fit the model:
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks=callbacks_list, validation_split= 0.2, 
                        verbose=1)
    print(model.summary())
    # visualize the training performance:
    show_performance(history)
    # save model:
    save_model(model, file_name, file_type)

    return model

def show_performance(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and  Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()


def save_model(model, file_name, file_type):

    if file_type == 'Keras':
        model.save(file_name +'.h5')
    elif file_type == 'JSON':
        # Serialize model to JSON
        model_json = model.to_json()
        with open(file_name + '.json', 'w') as json_file:
            json_file.write(model_json)
        ## serialize weights to HDF5
        model.save_weights(file_name + '_weights.h5')
    elif file_type == 'YAML':
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open(file_name + '.yaml', "w") as yaml_file:
            yaml_file.write(model_yaml)
         ## serialize weights to HDF5
        model.save_weights(file_name + '_weights.h5')
    
    else:
        print('ERROR: Could not save model with choosen file_type')
    
    print('Model saved')

def load_model(file_name, file_type):
    if file_type == 'Keras':
        model = keras.models.load_model(file_name+'.h5')
    elif file_type == 'JSON':
        json_file = open(file_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(file_name+'.h5')
    elif file_type == 'YAML':
        yaml_file = open(file_name + '.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = keras.models.model_from_yaml(loaded_model_yaml)
        model.load_weights(file_name+'.h5')
    else:
        print('ERROR: cloud not load model. Wrong file_type')
    print('Model loaded')
    return model



if __name__ == '__main__':
    # get daat
    data = get_data(file_path)
    # print(data.head(20))
    # show_info(data)
    # data.info()

########################################################################
#                  1) Pre-processing the data
## The data from WiSDM has a sampling rate of 20 Hz
## time periods of 80 -> 80/20 = 4 seconds per sample
## use this info to plot the accelerometer data
#######################################################################
    ## Need an encoded value for the dataframe
    LABEL = 'ActivityEncoded'
    le = preprocessing.LabelEncoder()
    data[LABEL] = le.fit_transform(data['activity'].values.ravel())
    LABELS = list(le.classes_)
    # show data for more insight:
    # # plot_activity(data)

#################################################################
#           2) split data into training and test sets:
# 
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
    data_train = normalize_data(data_train)
    data_test = normalize_data(data_test)

    # Preparing data for Keras
    ## Reshape data into segments of size TIME_PERIODS / sampling rate:
    ## 80/20 = 4 seconds intervals
    x_train, y_train = prepare_data_keras(data_train, TIME_PERIODS, STEP_DISTANCE, LABEL)
    # could also have used train_test_split:
    # X_train, X_test, y_train, y_test = train_test_split(
    #     reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

    # print('x_train shape: ', x_train.shape)
    # print(x_train.shape[0], 'training samples')
    # print('y_train shape: ', y_train.shape)
    # print(x_train)
    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    N_CLASSES = le.classes_.size
    # print(list(le.classes_))

    # The Keras only excepts a list of values so we have to transform our matrix of
    # 80x3 to a list of 240 values
    # TODO: remove this if not used
    # n_values = num_sensors *num_time_periods
    # x_train = x_train.reshape(x_train.shape[0], n_values)
    # keras only accept dataytpe float32:
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    #  conduct one-hot-encoding of our labels:
    y_train_hot = np_utils.to_categorical(y_train, N_CLASSES)

#############################################################################
#              3) Create model / estimator
# Here we create the  DNN and have to choose the amount of hidden
# layers and number of nodes. Now that the data is transformed and
# split we can also create a CNN or other complex NN if we want
#############################################################################
    # create create_model function
    N_NODES = 64
    N_LAYERS = 3
    COSTUME = True
    BATCH_SIZE = 200
    EPOCHS = 50
    file_name ="model_lstm"
    file_type ='Keras' # 'YAML' 'JSON
    # model_seq = create_DNN_model(N_NODES, N_LAYERS, COSTUME, N_CLASSES)
    # model_seq = create_LSTM_model(N_NODES, N_CLASSES)
    #model_seq = create_bidir_LSTM_model(N_NODES, N_CLASSES)
    #load model:
    model_seq = load_model(file_name, file_type)
    
   
##############################################################################
#           4) Train model and evaluate
#   Using a 80:20 split here between training and validation data
#   Only running this if not previously trained model
#  
##############################################################################
    #Train model:
    #model_seq = train_model(BATCH_SIZE, EPOCHS, file_name, file_type, model_seq, x_train, y_train_hot)
    print(model_seq.summary())

#############################################################################
#                   5) Evaluation and40 illustrations

# accuracy and loss, confusion matrix etc
#############################################################################

    # Print confusion matrix:
    y_pred_train = model_seq.predict(x_train)
    best_class_train = np.argmax(y_pred_train, axis=1)
    print(classification_report(y_train, best_class_train))

    # check against test data
    x_test, y_test = prepare_data_keras(data_test, TIME_PERIODS, STEP_DISTANCE, LABEL)
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    y_test = np_utils.to_categorical(y_test, N_CLASSES)
    score = model_seq.evaluate(x_test, y_test, verbose=1)
    print('\nAccuracy on test data: %0.2f' % score[1])
    print('\nLoss on test data: %0.2f' % score[0])
    40
    y_pred_test = model_seq.predict(x_test)
    best_class_pred_test = np.argmax(y_pred_test, axis=1)
    best_class_test = np.argmax(y_test, axis=1)


    create_confusion_matrix(best_class_test, best_class_pred_test, LABELS)
    print(classification_report(best_class_test, best_class_pred_test))
    print(LABELS)

