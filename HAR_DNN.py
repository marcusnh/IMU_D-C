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


def split_data(data, ratio=0.8):
    user_list =data.user_id.unique()
    # user_list = random.shuffle(user_list.tolist())
    user_list = user_list.tolist()
    random.shuffle(user_list)
    train_list = user_list[: int(len(user_list)*ratio)]
    test_list = user_list[int(len(user_list)*ratio):]
    data_train = data[data['user_id'].isin(train_list)]
    data_test = data[data['user_id'].isin(test_list)]
    return data_train, data_test

#function to count the number of classes
def _count_classes(y):
    return len(set([tuple(category) for category in y]))


# save and load functions:
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
        yaml_file = open(file_name + '.yaml', 'r')#
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = keras.models.model_from_yaml(loaded_model_yaml)
        model.load_weights(file_name+'.h5')
    else:
        print('ERROR: cloud not load model. Wrong file_type')
    print('Model loaded')
    return model

# Create models:

def DNN_model(n_neurons, n_hidden, learning_rate, n_classes=6):
    model = Sequential()
    model.add(Dense(n_neurons, activation='relu'))
    for i in range(n_hidden):
        model.add(Dense(n_neurons, activation='relu'))
        
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    
    model = build_model(model, learning_rate)

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
    model.add(LSTM(units = 128, input_shape =(TIME_PERIODS,N_FEATURES)))
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
    model.add(Dense(N_NODES, activation='relu'))
    model.add(Dropout(0.2,))
    model.add(Bidirectional(LSTM(units=128, #activation='relu',
                                return_sequences=True),
                                input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(Bidirectional(LSTM(round(60), activation='relu',
                                return_sequences=True),
                                input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(Dropout(0.5,))
    model.add(Dense(N_NODES, activation='relu'))

    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))


    return model

## train model:

def train_model(BATCH_SIZE, EPOCHS, file_name, file_type, model, x_train, y_train):
    callbacks_list = [
        # keras.callbacks.ModelCheckpoint(filepath='best_'+file_name+'.{epoch:02d}-{val_accuracy:.2f}.h5',
        #                                 monitor='accuracy', save_best_only=True), 
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1, patience=5, 
                                      min_delta=0.001),]

     # config the model  
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])
    # fit the model:
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks=callbacks_list, validation_split= 0.2, 
                        verbose=1)
    print(model.summary())
    # visualize the training performance:
    show_performance_DNN(history)
    # save model:
    save_model(model, file_name, file_type)#
    return model

def build_model(model, learning_rate=3e-3):
   
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer =optimizer, metrics = ['accuracy'])
    
    return model

def search_model(hp):
    model = Sequential()
    model.add(Flatten())
    for i in range(hp.Int("num_layers", 2, 20)):
        model.add(
            Dense(
                units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
    model.add(Dense(6, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
if __name__ == '__main__':
    # Extract data
    data = txt_to_pd_WISDM()
    ########################################################################
    #                  1) Pre-processing the data
    ########################################################################
    # file_path = 'Data/test/WISDM_feature.csv'
    # feature_df = pd.read_csv(file_path)
    # print(feature_df)
    # data = tilt_angle(data)
    # data['SVM'] = np.sqrt(data['x-axis']**2+data['y-axis']**2+data['z-axis']**2)
    # data = data[['user_id' , 'activity','timestamp','SVM']].copy()
    print(data)
    # Normalize data [-1,1]: across subjects or
    # data = normalize_data(data)
    
    # #FILTERING:
    # data = filter_data(data,fs_share=0.45, nr_medfil=3)

    
    #################################################################
    #           2) split data into training and test sets:
    # 
    # Idea here: let our model learn from a couple of people doing
    # all the different activity. We could also have taken some data
    # from every person and tested on the remaining data, since some 
    # movment is different from person to person, but we consider this
    # to be negligible. This has t o be considered becuase it will effect
    # the performance of the DNN
    ##################################################################
    segment_array, segment_labels,LABELS = extract_windows(data, sec=2.5, overlap_prosent=50)
    print(segment_array.shape)
    # data_train, data_test = split_data(data, ratio=0.8)
    # #  ID 1-28 for training and 28>for testing
    # data_test = data[data['user_id'] > 28].copy()
    # data_train = data[data['user_id'] <= 28].copy()
    # X_test, y_test, LABELS = extract_windows(data_test, sec=2.5, overlap_prosent=20)
    # X_train, y_train, LABELS = extract_windows(data_train, sec=2.5, overlap_prosent=20)
    random_seed=42
    X_train, X_test, y_train, y_test = train_test_split(segment_array, 
        segment_labels, test_size = 0.2, random_state = random_seed)
        
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
                                    test_size=0.5, random_state=1) 
    # Conduct one-hot-encoding of labels:
    y_train_hot = pd.get_dummies(y_train).to_numpy()
    y_test_hot = pd.get_dummies(y_test).to_numpy()
    y_val_hot = pd.get_dummies(y_val).to_numpy()
        
    timesteps = len(X_train[0])
    input_dim = len(X_train[0][0])
    n_classes = _count_classes(y_train_hot)

    print(len(X_train), timesteps, input_dim)

    print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
    print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))
    print('X_val  and y_val  : ({},{})'.format(X_val.shape, y_val.shape))


    #############################################################################
    #              3/4) Create and train model / Load exsisting model
    # Here we create the  DNN and have to choose the amount of hidden
    # layers and number of nodes. Now that the data is transformed and
    # split we can also create a CNN or other complex NN if we want
    #############################################################################
    # Initializing parameters
    n_neurons = 50
    batch_size = 1024
    n_hidden = 3
    learning_rate = 0.0025
    l2_loss = 0.0015
    COSTUME = True
    file_name ="model_feature3_bidir_LTSM"
    file_type ='Keras' # 'YAML' 'JSON

    keras_param_space = {"n_hidden": [1, 2, 3, 4],
                      "n_neurons": [50], #np.arange(30, 300),
                      "learning_rate":[3e-4, 3e-2],
                      }

    model = DNN_model(n_neurons, n_hidden, n_classes)
    # model= LSTM_model(n_neurons, n_classes, timesteps, N_FEATURES=input_dim)
    # model = bidir_LSTM_model(n_neurons, n_classes, timesteps, N_FEATURES=input_dim)
    # keras_clf = tf.keras.wrappers.scikit_learn.KerasClassifier(DNN_model)
    # model = GridSearchCV(keras_clf, keras_param_space, #n_iter=20, 
    #                                cv=5, scoring="accuracy", n_jobs=6, verbose=True)
    # tuner = RandomSearch(search_model, 
    #                         objective='val_accuracy', 
    #                         max_trials=3,
    #                         executions_per_trial=2,
    #                         overwrite=True)

    # tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    # model = tuner.get_best_models(num_models=2)
    # print(tuner.results_summary())
    # history = model.fit(X_train, y_train, epochs=1,
    #                    validation_data=(X_val, y_val),
    #                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    # # train model:
    model = train_model(batch_size, n_neurons, file_name, file_type, model, X_train, y_train_hot)
    performance = {}
    performance['model'] = model
    performance['best_params_'] = model.best_params_
    # print(model.summary())
    print('\n\n==> Best Estimator:')
    print('\t{}\n'.format(model.best_estimator_))
    # parameters that gave best results while performing grid search
    print('\n==> Best parameters:')
    print('\tParameters of best estimator : {}'.format(model.best_params_))


    #load model:
    # model = load_model(file_name, file_type)
   
    # validation data performance
    y_pred_val = model.predict(X_val)
    
    best_class_val = np.argmax(y_pred_val, axis=1)
    
    print('Classification report for training data:')
    print(classification_report(y_val, best_class_val))
    # Evaluation score: categorical cross-entropy and accuracy
    score = model.evaluate(X_val, y_val_hot)
    performance['validation score'] = score
    # Test data performance
    y_pred_test = model.predict(X_test)
    best_class_pred_test = np.argmax(y_pred_test, axis=1)
    best_class_test = np.argmax(y_test_hot, axis=1)
    
    print('Classification report for test data')
    print(classification_report(best_class_test, best_class_pred_test))
    confusion_matrix(best_class_test, best_class_pred_test, LABELS, normalize=True)
    
    # Evaluation score: categorical cross-entropy and accuracy
    score = model.evaluate(X_test, y_test_hot)
    performance['Test score'] = score
   

    print("\n   cat_crossentropy  ||   accuracy ")
    print("____________________________________")
    print(score)
    t = PrettyTable(['NN model', 'cat_crossentropy', 'Accuracy','model_param'])
    t.add_row([file_name,score[0],score[1], model.best_params_])
    with open('HAR_DNN.txt', 'a+') as f:
        f.write(str(t))