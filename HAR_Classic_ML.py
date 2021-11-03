
##################################################################################
#  Classical machine larning algorithmes:
#   Using ML-fuctions from Sklearn:
#   Linear SVC, SVC, decision tree, random forest and gradient Boosting classifier

###################################################################################

import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from prettytable import PrettyTable

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform
from skopt import BayesSearchCV

from get_data import txt_to_pd_WISDM, get_UCI_data
from feature_extraction import feature_extraction
from pre_processing import split_train_test_data, normalize_data, filter_data
from visualize_data import confusion_matrix

def execute_model(model, X_train, y_train, X_test, 
                 y_test, class_labels, normalize=True):
    # Store model result in dict:
    performance = {}
    # start training model:
    print('Start training model:')
    training_start = datetime.now()
    model.fit(X_train, y_train)
    training_end = datetime.now()
    print('Done training model!')
    performance['training_time'] = training_end - training_start
    print('Total training time: {}'.format(performance['training_time']))

    # Test model:
    print(' Testing model on test data:')
    test_start = datetime.now()
    y_pred = model.predict(X_test)
    test_end = datetime.now()
    print('Done testing model!')
    performance['test_time'] = test_end - test_start
    print('Total test time: {}'.format(performance['test_time']))
    performance['predicted'] = y_pred

    # Show model performance stats:
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    performance['accuracy'] = accuracy
    print('==> Accuracy:- {}\n'.format(accuracy))
    # confusion matrix:
    # confusion_matrix(y_test, y_pred, class_labels, normalize=normalize) 
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # classification report:
    print('********Classification Report********')
    classification_report = metrics.classification_report(y_test, y_pred)
    print(classification_report)
    performance['classification_report'] = classification_report
    # Store model:
    performance['model'] = model

    return performance



def model_tuning(model, class_labels,X_train, y_train, X_test, 
                 y_test, grid_param, n_jobs, cv=None, verbose=0, ):
    # performing grid search to find the best parameters to give most accurate predictions
    # . The value of the hyperparameter has to be set before the learning process begins
    # For example, c in Support Vector Machines, k in k-Nearest Neighbors,
    #  the number of hidden layers in Neural Networks.
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('Starting grid search for new model:')
    # search = BayesSearchCV(estimator=model, search_spaces=grid_param, n_jobs=-1, cv=cv)
    search = GridSearchCV(model, param_grid=grid_param,  n_jobs= n_jobs,
                             cv=cv, verbose=verbose,scoring='accuracy')
    # search = RandomizedSearchCV(model, grid_param, n_iter=500, scoring='accuracy', n_jobs=n_jobs, cv=cv, random_state=1)
    # search = BayesSearchCV(model, grid_param, n_iter=32, # specify how many iterations
                                    # scoring="accuracy", n_jobs=-1, cv=5)

    # search.fit(X, y)
    # # report the best result
    # print(search.best_score_)
    # print(search.best_params_)
    
    model_perf = execute_model(search, X_train, y_train, X_test, 
                 y_test, class_labels, normalize=True)
    performance = model_perf['model']

    print('\n\n==> Best Estimator:')
    print('\t{}\n'.format(performance.best_estimator_))


    # parameters that gave best results while performing grid search
    print('\n==> Best parameters:')
    print('\tParameters of best estimator : {}'.format(performance.best_params_))


    #  number of cross validation splits
    print('\n==> No. of CrossValidation sets:')
    print('\tTotal numbre of cross validation sets: {}'.format(performance.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('\n==> Best Score:')
    print('\tAverage Cross Validate scores of best estimator : {}'.format(performance.best_score_))
    return model_perf

if __name__ == '__main__':
    #Loading data:
    data = txt_to_pd_WISDM()
    # data = normalize_data(data)
    # data = filter_data(data,fs_share=0.45, nr_medfil=5)
    # feature_df = feature_extraction(data, sec=5.5, overlap_prosent=50)
    file_path = 'Data/test/WISDM_feature_new.csv'
    # feature_df.to_csv(file_path)
    # file_path = 'Data/test/arff_test.csv'
    feature_df = pd.read_csv(file_path, index_col=0 )
    print(feature_df)
    # split data:
    random_seed =42
    X_train, X_test, X_val, y_train, y_test, y_val,class_labels = \
    split_train_test_data(feature_df, ratio=0.8)
    #  conduct one-hot-encoding of our labels:
    y_train_hot = pd.get_dummies(y_train).to_numpy()
    y_test_hot = pd.get_dummies(y_test).to_numpy()
    y_val_hot = pd.get_dummies(y_train).to_numpy()
    

    # #TEST WITH UCI data:
    # X_train, X_test, y_train, y_test, class_labels= get_UCI_data()

    # print(test_df.head(2))
    print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
    print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))
    print('X_val  and y_val  : ({},{})'.format(X_val.shape, y_val.shape))
    # class_labels = {'Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'} 
    
    ######################################################
    # Test clasical machine learning methods
    ## Logistic regression:
    # define evaluation
    log_reg = linear_model.LogisticRegression(solver='liblinear')
    params = {'C':[0.01, 0.1, 1, 10, 20, 30], 
                    'penalty':['l2','l1']}

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
    print ('cv: ', cv)
    n_jobs = -1 # automatically use all of the cores in your system
    
    # perform the search
    parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 
                    'penalty':['l2','l1']}

    log_reg_res = model_tuning(log_reg,class_labels, X_train, y_train, X_val, 
                 y_val, parameters, n_jobs=n_jobs, cv=cv, verbose=1)
    
    # ## Linear support vector classification
    parameters = {'C':[0.125, 0.5, 1, 2, 8, 16, 30]}
    lr_svc = LinearSVC(tol=0.00005, dual=False)
    lr_svc_res = model_tuning(lr_svc,class_labels, X_train, y_train, X_test, 
                 y_test, parameters, n_jobs=-1, verbose=1)
    ## kernel SVM:
    kernel_scv = SVC(kernel='rbf', gamma ='scale')
    parameters = {'C':[2,8,16,32,100,500,1000],\
                 'gamma': [ 0.0078125, 0.125, 2]}
    kernel_scv_res = model_tuning(kernel_scv,class_labels, X_train, y_train, X_test, 
                 y_test, parameters, n_jobs=-1,)
    
    ## decision tree classifier:
    parameters = {'max_depth': np.arange(1,10,1)}
    dtc = DecisionTreeClassifier()
    dtc_res = model_tuning(dtc,class_labels, X_train, y_train, X_test, 
                 y_test, parameters, n_jobs=-1,)
    # # Random Forest Classifier
    params = {'n_estimators': np.arange(1,181,25), 
                'max_depth': np.arange(1,25,2)}
    rfc = RandomForestClassifier()
    rfc_res = model_tuning(rfc,class_labels, X_train, y_train, X_test, 
                 y_test, params, n_jobs=-1,)

    # Gradient Boosted Decision Trees:
    params = {'max_depth': [1,4,6,26],
              'n_estimators': np.arange(1,20)}
    gbdt = GradientBoostingClassifier()
    gbdt_res = model_tuning(gbdt,class_labels, X_train, y_train, X_test, 
                 y_test, params, n_jobs=-1,)

    # save models:             
    model_res ={'Logistic Regression':log_reg_res, 
                'Linear SVC ': lr_svc_res,
               'rbf SVM classifier': kernel_scv_res,
                ' Decision tree classifier' :dtc_res, 
                'Random Forest classifier':rfc_res, 
                'Gradient Boosted Decision Trees': gbdt_res}
    
    print('\n      ML model      Accuracy     Error')
    print('   ---------------------------------------')
    t = PrettyTable(['ML model', 'Accuracy', 'Error', 'Best model parameters', 'Training Time', 'Test Time'])
    for key in model_res:
        accuracy = str(round(model_res[key]['accuracy']*100, 2))+'%'
        error = str(round(100-(model_res[key]['accuracy']*100), 2)) +'%'
        model_params = model_res[key]['model'].best_params_
        training_t = model_res[key]['training_time']
        test_t = model_res[key]['test_time']
        t.add_row([key, accuracy, error, model_params, training_t, test_t])
    print(t)
    with open('HAR_classic_ML.txt', 'a+') as f:
        f.write(str(t))
    
    # Performance on test set:
    print('Performance on test set:')
    for key in model_res:
        print('model: ', key)
        model = model_res[key]['model']
        y_pred = model.predict(X_test)
        # Show model performance stats:
        accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        print('==> Accuracy:- {}\n'.format(accuracy))
        cm = metrics.confusion_matrix(y_test, y_pred)
        print('********Classification Report********')
        classification_report = metrics.classification_report(y_test, y_pred)
        print(classification_report)
        #confusion matrix:
        confusion_matrix(y_test, y_pred, class_labels, normalize=True)

