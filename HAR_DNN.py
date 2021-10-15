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
from prettytable import PrettyTable

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Bidirectional
from keras.regularizers import L1L2
from keras.utils import np_utils

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

def DNN_model(N_NODES, N_LAYERS, COSTUME, N_CLASSES):
    model = Sequential()
    if (COSTUME):
        model.add(Dense(N_NODES*2, activation='relu'))
        model.add(Dense(N_NODES, activation='relu'))
        model.add(Dense(N_NODES, activation='relu'))
        
        model.add(Dense(N_NODES, activation='relu'))
        model.add(Flatten())
        model.add(Dense(N_CLASSES, activation='softmax'))
    else:
        NODES = random.randint(N_NODES/2, N_NODES*2)
        for i in range(N_LAYERS):
            model.add(Dense(N_NODES, activation='relu'))
            
        model.add(Flatten())
        model.add(Dense(N_CLASSES, activation='softmax'))

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
    #RNN
    # Bias regularizer value - we will use elasticnet
    reg = L1L2(0.01, 0.01)
    model = Sequential()
    model.add(Dense(N_NODES*2, activation='relu'))
    model.add(Dense(N_NODES, activation='relu'))
    model.add(LSTM(64, activation='relu',input_shape=(TIME_PERIODS,N_FEATURES), 
                    return_sequences=True,bias_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='sigmoid'))
    return model

def bidir_LSTM_model(N_NODES, N_CLASSES,TIME_PERIODS, N_FEATURES):
    #BRNN
    model = Sequential()
    model.add(Dense(N_NODES, activation='relu'))
    model.add(Dense(N_NODES, activation='relu'))
    model.add(Dropout(0.2,))
    model.add(Bidirectional(LSTM(round(64), #activation='relu',
                                return_sequences=True),
                                input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(Bidirectional(LSTM(round(30), activation='relu',
                                return_sequences=True),
                                input_shape=(TIME_PERIODS,N_FEATURES)))
    model.add(Dropout(0.2,))
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))


    return model

## train model:

def train_model(BATCH_SIZE, EPOCHS, file_name, file_type, model, x_train, y_train):
    callbacks_list = [
        # keras.callbacks.ModelCheckpoint(filepath='best_'+file_name+'.{epoch:02d}-{val_accuracy:.2f}.h5',
        #                                 monitor='accuracy', save_best_only=True), 
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',verbose=1, patience=12, 
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

if __name__ == '__main__':
    # Extract data
    data = txt_to_pd_WISDM()
    ########################################################################
    #                  1) Pre-processing the data
    ########################################################################
    file_path = 'Data/test/WISDM_feature.csv'
    feature_df = pd.read_csv(file_path)
    print(feature_df)
    # data = tilt_angle(data)
    # data['SVM'] = np.sqrt(data['x-axis']**2+data['y-axis']**2+data['z-axis']**2)
    # Normalize data [-1,1]: across subjects or
    # data = normalize_data(data)
    # print(data)
    
    # #FILTERING:
    # data = filter_data(data)

    
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
    data_train, data_test = split_data(data, ratio=0.9)
    # #  ID 1-28 for training and 28>for testing
    # data_test = data[data['user_id'] > 28].copy()
    # data_train = data[data['user_id'] <= 28].copy()
    X_test, y_test, LABELS = extract_windows(data_test, sec=6.4, overlap_prosent=50)
    X_train, y_train, LABELS = extract_windows(data_train, sec=6.4, overlap_prosent=50)
     #  conduct one-hot-encoding of our labels:
    y_test_hot, y_train_hot = pd.get_dummies(y_test).to_numpy(), pd.get_dummies(y_train).to_numpy()

    timesteps = len(X_train[0])
    input_dim = len(X_train[0][0])
    n_classes = _count_classes(y_train_hot)

    print(timesteps)
    print(input_dim)
    print(len(X_train))

    #############################################################################
    #              3/4) Create and train model / Load exsisting model
    # Here we create the  DNN and have to choose the amount of hidden
    # layers and number of nodes. Now that the data is transformed and
    # split we can also create a CNN or other complex NN if we want
    #############################################################################
    # Initializing parameters
    epochs = 30
    batch_size = 16
    n_hidden = 32
    N_LAYERS = 3
    COSTUME = True
    file_name ="model_feature2_yz_LTSM"
    file_type ='Keras' # 'YAML' 'JSON
    # model = DNN_model(epochs, N_LAYERS, COSTUME, n_classes)
    model= LSTM_model(epochs, n_classes, timesteps, N_FEATURES=input_dim)
    # model = bidir_LSTM_model(epochs, n_classes, timesteps, N_FEATURES=input_dim)
    # train model:
    model = train_model(batch_size, epochs, file_name, file_type, model, X_train, y_train_hot)

    #load model:
    # model = load_model(file_name, file_type)
   

    # Print confusion matrix:
    y_pred_train = model.predict(X_train)
    best_class_train = np.argmax(y_pred_train, axis=1)
    print('Classification report for training data:')
    print(classification_report(y_train, best_class_train))
   

    y_pred_test = model.predict(X_test)
    best_class_pred_test = np.argmax(y_pred_test, axis=1)
    best_class_test = np.argmax(y_test_hot, axis=1)
    confusion_matrix(best_class_test, best_class_pred_test, LABELS, normalize=True)
    print('Classification report for test data')
    print(classification_report(best_class_test, best_class_pred_test))
    print(LABELS)

    score = model.evaluate(X_test, y_test_hot)
    # # repeat experiment
    # scores = list()
    # for r in range(repeats):
    #     score = evaluate_model(X_train, y_train, X_test, testy)
    #     score = score * 100.0
    #     print('>#%d: %.3f' % (r+1, score))
    #     scores.append(score)

    print("\n   cat_crossentropy  ||   accuracy ")
    print("____________________________________")
    print(score)
    t = PrettyTable(['NN model', 'cat_crossentropy', 'Accuracy',])
    t.add_row([file_name,score[0],score[1]])
    with open('HAR_DNN.txt', 'a+') as f:
        f.write(str(t))