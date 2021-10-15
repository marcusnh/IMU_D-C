import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn import metrics

from get_data import txt_to_pd_WISDM



def total_activities(data):
    sns.set_style('whitegrid')
    # data['activity'].value_counts().plot(kind='bar', title='Number of acitivity samples')
    sns.countplot(x='activity', data=data)
    plt.show()


def activity_data_per_user(data):
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Dejavu Sans'

    plt.figure(figsize=(16,8))
    plt.title('Data provided by each user', fontsize=20)
    sns.countplot(x='user_id',hue='activity', data = data)
    plt.show()

def activity_wise_dist(data, column):
    sns.set_palette("Set1", desat=0.80)
    facetgrid = sns.FacetGrid(data, hue='activity')
    facetgrid.map(sns.kdeplot, column ).add_legend()
    plt.show()


def activity_boxplot_dist(data, column):
    #group magnitude data
    sns.set_palette("Set1", desat=0.80)
    # facetgrid = sns.FacetGrid(data, hue='activity')
    # facetgrid.map(sns.kdeplot, column ).add_legend()
    # plt.show()
    #Show boxplot:
    plt.figure(figsize=(7,5))
    sns.boxplot(x='activity', y=column,data=data, showfliers=False, saturation=1)
    plt.ylabel(column +'')
    # plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
    # plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
    plt.xticks(rotation=40)
    plt.show()

def show_activity(data, activity, user_id, samples=128):
    # Show single activity for a spesific user
    user_data = data[data['user_id']==user_id]
    user_data = user_data.drop(columns=['timestamp', 'user_id'])
    activities = user_data['activity'].unique()
    print('\nNumber of samples per activity:')
    print(user_data['activity'].value_counts())
    user_data = user_data[user_data['activity']==activity]
    title = activity+' for user: ' +str(user_id)
    user_data[0:samples].plot(title=title)
    plt.show()



def compare_user_activitys(data, user_id, samples=128):
    #Look at all activities of user with pre defined number of samples
    user_data = data[data['user_id']==user_id]
    user_data = user_data.drop(columns=['timestamp', 'user_id'])
    
    activities = user_data['activity'].unique()
    print('\nNumber of samples per activity:')
    print(user_data['activity'].value_counts())

    fig, axes = plt.subplots(nrows=1, ncols=len(activities), figsize=(10, 5))
    counter = 0
    # user_data[user_data['activity'] == 'Jogging'].to_csv(path_or_buf='testfile.csv')
    for i in activities:
        activity_data = user_data[user_data['activity'] == i]
        activity_data.index = range(0,len(activity_data))
        activity_data[0:samples].plot(title=i, ax=axes[counter])
        counter +=1
    
    plt.show()

def activity_difference_between_users(data, users, activity):
    cnt = 0
    fig, axes = plt.subplots(nrows=1, ncols=len(users), figsize=(20, 10))
    plt.suptitle('Comparing activity:'+activity)
    for i in users:
        user_data =data[(data['user_id'] ==i) & (data['activity'] == activity)]
        user_data.index = range(0,len(user_data))
        # if user_data.empty:
        #     print(user_data)

        user_data = user_data.drop(columns=['timestamp', 'user_id'])
        user_data.plot(title='User id: ' +str(i), ax=axes[cnt])
        cnt += 1

    plt.show()



def confusion_matrix(vali, predict, LABELS, normalize=False):
    matrix = metrics.confusion_matrix(vali, predict) 
    print('\n ********Confusion Matrix********')
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalize else 'd'
    plt.figure( figsize=(6, 4))
    sns.heatmap(matrix, cmap='coolwarm', linecolor='white',linewidths=1, 
                xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=fmt)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def show_performance_DNN(history):
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

def execute_TSNE(data, perplexities=[2,5,10,20,50], n_iter=1000, 
                 img_name_prefix='t-sne'):
    y_data = data['activity']
    # print(y_data)
    X_data = data.drop(['user_id', 'activity'],axis=1)
    for i, perplexity in enumerate(perplexities):
        # create new x iwth lower dimensions:
        X_new= TSNE(verbose=2, n_iter=n_iter,
                         perplexity=perplexity).fit_transform(X_data)
        # new dataframe to create plot:
        df = pd.DataFrame({'x':X_new[:,0], 'y':X_new[:,1],
                           'label':y_data})
        sns.lmplot(data=df, x='x', y='y', hue='label',\
                   fit_reg=False, height=8, palette="Set1",)
                #    markers=['^','P','8','s', 'o','*', 'p'])
        plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
        img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
        print('saving this plot as image in present working directory...')
        plt.savefig(img_name)
        plt.show()
        print('Done with performing tsne with perplexity {} and with {} iterations at max'.format(perplexity, n_iter))
    return 0


if __name__ == '__main__':
    file_path = 'Data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    data = txt_to_pd_WISDM(file_path)

    # data['mean'] = data[['x-axis', 'y-axis', 'z-axis']].mean(numeric_only=True, axis=1)
    print(data)
    # total_activities(data)
    # activity_data_per_user(data)
    users = [1, 2, 33, 29]
    activity = 'Downstairs'
    # compare_user_activitys(data , users[0])
    activity_difference_between_users(data, users, activity)

    # data = normalize_data(data)
    # # data = tilt_angle(data)
    # # data = SVM(data)
    # print(data)

    # # create plots to se patterns in data:
    # # print(data[['activity','SVM']])
    # activity_data_per_user(data)
    # column = 'z-axis'
    # activity_wise_dist(data, column)
    # activity_boxplot_dist(data, column)
    # execute_TSNE(data[0:10000],)

