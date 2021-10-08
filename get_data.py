import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 

# import arff

file_path = 'Data/WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff'

def create_float(value):
    try:
        return np.float(value)
    except:
        return np.nan

def txt_to_pd_WISDM(file_path='Data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'):
    columns = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    #extract data from file_path:
    data = pd.read_csv(file_path, header=None, names=columns)
    # Last column has a ; that needs to be removed and transformed to float
    data['z-axis'].replace(regex=True, inplace=True,to_replace=r';', value=r'')
    data['z-axis'] = data['z-axis'].apply(create_float)
    #drop null values:
    data.dropna(axis=0, how='any',inplace=True)
    #drop the rows where the timestamp is 0:
    data = data[data['timestamp'] != 0]
    # now arrange data in ascending order of the user and timestamp
    data = data.sort_values(by = ['user_id', 'timestamp'], ignore_index=True)
    return data

def arff_to_pd_WISDM(file_path='Data/WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff'):
    col_names = {'user': 'user_id', 'class':'activity', 
                 'RESULTANT': 'SVM'}
    col_keep = ['user', 'class', 'XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 
                'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
                'XSTANDDEV','YSTANDDEV', 'ZSTANDDEV', 
                'RESULTANT']
    data, meta = loadarff(file_path)
    print(data['XAVG'])
    df_data = pd.DataFrame(data)
    df_data = df_data[col_keep]
    df_data['user'] = df_data['user'].str.decode('utf-8')
    df_data['class'] = df_data['class'].str.decode('utf-8')
    df_data = df_data.rename(columns=col_names)
    #  drop null values:
    df_data.dropna(axis=0, how='any',inplace=True)
    # now arrange data in ascending order of the user and timestamp
    df_data = df_data.sort_values(by = ['user_id'], ignore_index=True)
    return df_data

def get_UCI_data():
    features = list()
    with open('Data/UCI/UCI_HAR_Dataset/features.txt') as f:
        features = [line.split()[1] for line in f.readlines()]
    print('No of Features: {}'.format(len(features)))

    # get the data from txt files to pandas dataffame
    X_train = pd.read_csv('Data/UCI/UCI_HAR_Dataset/train/X_train.txt', delim_whitespace=True, header=None)
    X_train.columns = [features]
    # add subject column to the dataframe
    X_train['subject'] = pd.read_csv('Data/UCI/UCI_HAR_Dataset/train/subject_train.txt', header=None, squeeze=True)

    y_train = pd.read_csv('Data/UCI/UCI_HAR_Dataset/train/y_train.txt', names=['Activity'], squeeze=True)
    y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                        4:'SITTING', 5:'STANDING',6:'LAYING'})

    # put all columns in a single dataframe
    train = X_train
    train['Activity'] = y_train
    train['ActivityName'] = y_train_labels
    train.to_csv(path_or_buf='Data/features_test.csv')


    # get the data from txt files to pandas dataffame
    X_test = pd.read_csv('Data/UCI/UCI_HAR_Dataset/test/X_test.txt', delim_whitespace=True, header=None)
    X_test.columns = [features]
    # add subject column to the dataframe
    X_test['subject'] = pd.read_csv('Data/UCI/UCI_HAR_Dataset/test/subject_test.txt', header=None, squeeze=True)

    # get y labels from the txt file
    y_test = pd.read_csv('Data/UCI/UCI_HAR_Dataset/test/y_test.txt', names=['Activity'], squeeze=True)
    
    y_test_labels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',\
                           4:'SITTING', 5:'STANDING',6:'LAYING'})


    # put all columns in a single dataframe
    test = X_test
    test['Activity'] = y_test
    test['ActivityName'] = y_test_labels
    test.sample(2)

    # print('No of duplicates in train: {}'.format(sum(train.duplicated())))
    # print('No of duplicates in test : {}'.format(sum(test.duplicated())))

    # print('We have {} NaN/Null values in train'.format(train.isnull().values.sum()))
    # print('We have {} NaN/Null values in test'.format(test.isnull().values.sum()))

    train.to_csv('Data/features_train.csv', index=False)
    test.to_csv('Data/features_test.csv', index=False)

    train = pd.read_csv('Data/features_train.csv')
    test = pd.read_csv('Data/features_test.csv')
    X_test = X_test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    X_train = X_train.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    class_labels =['LAYING', 'SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
    return  X_train, X_test, y_train_labels, y_test_labels, class_labels


   
if __name__ == '__main__':
    
    # data = txt_to_pd_WISDM()
    # Get data from WISDOM_arff:
    # data = arff_to_pd_WISDM(file_path)
    # testfolder = 'Data/test/arff_test.csv'
    # data.to_csv(testfolder)
    # file_path = 'Data/test/arff_test.csv'
    # feature_df = pd.read_csv(file_path, index_col=0 )
    # print(feature_df)

    X_train, X_test, y_train, y_test, class_labels= get_UCI_data()
    print(X_train ,y_train)

