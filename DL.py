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
from scipy.stats import reciprocal

from IPython.display import display, HTML
from prettytable import PrettyTable

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from skopt import BayesSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Bidirectional
from keras.regularizers import L1L2
from keras.utils import np_utils
from keras_tuner import RandomSearch

import tensorflow as tf

from get_data import txt_to_pd_WISDM
from feature_extraction import tilt_angle
from pre_processing import extract_windows, normalize_data, filter_data
from visualize_data import confusion_matrix, show_performance_DNN
#function to count the number of classes
def _count_classes(y):
    return len(set([tuple(category) for category in y]))

# Create models:

def DNN_model(n_neurons, n_hidden, learning_rate, n_classes=6):
    model = Sequential()
    model.add(Dense(n_neurons, activation='relu'))
    for i in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    return model

def CNN_model(N_NODES, N_CLASSES, TIME_PERIODS, N_FEATURES):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model

def LSTM_model(N_NODES, N_CLASSES, TIME_PERIODS, N_FEATURES):
    model = Sequential()
    # RNN layer
    # Bias regularizer value - we will use elasticnet
    model.add(LSTM(units =60, return_sequences=True,
                   input_shape =(TIME_PERIODS,N_FEATURES)))
    model.add(LSTM(units = 60, return_sequences=True,
                     input_shape =(TIME_PERIODS,N_FEATURES)))
    # Dropout layer
    model.add(Dropout(0.5)) 
    # Dense layer with ReLu
    model.add(Dense(units = 64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))

    return model

def bidir_LSTM_model(N_NODES, N_CLASSES,TIME_PERIODS, N_FEATURES):
    #BRNN
    model = Sequential()
    # model.add(Dense(N_NODES, activation='relu'))
    # model.add(Dropout(0.2,))
    model.add(Bidirectional(LSTM(units=60, #activation='relu',
                                return_sequences=True),
                                input_shape=(TIME_PERIODS,N_FEATURES)))
    # model.add(Bidirectional(LSTM(round(60), activation='relu',
    #                             return_sequences=True),
    #                             input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(Dropout(0.5,))
    model.add(Dense(N_NODES, activation='relu'))

    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))


    return model


random_seed=42

 # Extract data
data = txt_to_pd_WISDM()
# Normalize data [-1,1]: across subjects or
# data = normalize_data(data)

segment_array, segment_labels,LABELS = extract_windows(data, sec=5.5, overlap_prosent=50)
print(segment_array.shape)
# 80% trening, 10% validering, 10% test 
X_train, X_test, y_train, y_test = train_test_split(segment_array, 
        segment_labels, test_size = 0.1, random_state = random_seed)



 # Conduct one-hot-encoding of labels:
y_train_hot = pd.get_dummies(y_train).to_numpy()
y_test_hot = pd.get_dummies(y_test).to_numpy()
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(y_train_hot)

print('Size of training data:')
print(len(X_train), timesteps, input_dim)
print('HERE!')
print(n_classes)
# Initializing parameters:
learning_rate = 0.0001
N_NODES = 40
N_HL = 4
epochs = 40 # one pass over the batch-size dataset
batch_size =  100 #1024
# Create model
model = DNN_model(N_NODES, N_HL, n_classes)
# model= LSTM_model(N_NODES, n_classes, timesteps, N_FEATURES=input_dim)
# model = bidir_LSTM_model(N_NODES, n_classes, timesteps, N_FEATURES=input_dim)

# Train model:
callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath='best_model_'+'.{epoch:02d}-{val_accuracy:.2f}.h5',
                                         monitor='accuracy',mode='max', save_best_only=True), 
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',restore_best_weights=True,
                                      verbose=1, patience=5, min_delta=0.001),]
 # config the model  
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer =opt, metrics = ['accuracy'])
# fit the model:
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    callbacks=callbacks_list, validation_split= 0.1111, 
                    verbose=1)
print(model.summary())
# visualize the training performance:
show_performance_DNN(history)
performance = {}
performance['model'] = model

# Evaluate score on test data: (ONLY AFTER TRAINING MODEL!!!)
# Test data performance
y_pred_test = model.predict(X_test)
best_class_pred_test = np.argmax(y_pred_test, axis=1)
best_class_test = np.argmax(y_test_hot, axis=1)

print('Classification report for test data')
print(classification_report(best_class_test, best_class_pred_test))
confusion_matrix(best_class_test, best_class_pred_test, LABELS, normalize=True)

# Evaluation score: categorical cross-entropy and accuracy
print(X_test.shape,y_test_hot.shape)
score = model.evaluate(X_test, y_test)
performance['Test score'] = score