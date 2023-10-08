import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from ML_utils import balance_data
import pickle
from sklearn.model_selection import train_test_split



def init_directories(w_path, plots_path, mod_path, np_arr_path):
    if not os.path.exists("./" + plots_path):
        os.mkdir("./" + plots_path)
    if not os.path.exists("./" + plots_path + "/anova_avg/"):
        os.mkdir("./" + plots_path + "/anova_avg/")
    if not os.path.exists("./" + plots_path + "/anova_avg/som_bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_avg/som_bal_comp")
    if not os.path.exists("./" + plots_path + "/anova_avg/som_no-bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_avg/som_no-bal_comp")
    if not os.path.exists("./" + plots_path + "/anova_min/"):
        os.mkdir("./" + plots_path + "/anova_min/")
    if not os.path.exists("./" + plots_path + "/anova_min/som_bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_min/som_bal_comp")
    if not os.path.exists("./" + plots_path + "/anova_min/som_no-bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_min/som_no-bal_comp")
    if not os.path.exists("./" + mod_path):
        os.mkdir("./" + mod_path)
    if not os.path.exists("./" + mod_path + "/anova_avg/"):
        os.mkdir("./" + mod_path + "/anova_avg/")
    if not os.path.exists("./" + mod_path + "/anova_min/"):
        os.mkdir("./" + mod_path + "/anova_min/")
    if not os.path.exists("./" + np_arr_path):
        os.mkdir("./" + np_arr_path)
    if not os.path.exists("./" + np_arr_path + "/anova_avg/"):
        os.mkdir("./" + np_arr_path + "/anova_avg/")
    if not os.path.exists("./" + np_arr_path + "/anova_min/"):
        os.mkdir("./" + np_arr_path + "/anova_min/")
    if not os.path.exists("./UCI HAR Dataset split/"):
        os.mkdir("./UCI HAR Dataset split/")
    if not os.path.exists("./UCI HAR Dataset split/train/"):
        os.mkdir("./UCI HAR Dataset split/train/")
    if not os.path.exists("./UCI HAR Dataset split/test"):
        os.mkdir("./UCI HAR Dataset split/test/")


# load a file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.to_numpy()


def load_group(filename, numFeat, pathPrefix=""):
    loaded = list()
    for name in filename:
        data = load_file(pathPrefix + name)
        loaded.append(data)

    print("loaded before", loaded)
    loaded = np.array(loaded[0][:, :numFeat])
    print("loaded after", loaded.shape)

    return loaded


def load_dataset_group(group, pathPrefix, numFeat):
    filepath = pathPrefix + group + "/"

    filename = list()
    # The “X_train.txt” file that contains the engineered features intended for fitting a model.
    filename += ["X_" + group + ".txt"]

    X = load_group(filename, numFeat, filepath)
    # The “y_train.txt” that contains the class labels for each observation (1-6).
    y = load_file(filepath + "/y_" + group + ".txt")

    return X, y

def load_subject_dataset(subject_index, uci_trainX, uci_trainy, sub_map):
    train_subjects = np.unique(sub_map)
    datasetX, datasety = dataset_for_subject(uci_trainX, uci_trainy, sub_map, train_subjects[subject_index])
    
    # split subject dataset in 70% train and 30% test
    trainX, testX, trainy, testy = train_test_split(datasetX, datasety, train_size=0.70, shuffle=True)

    print("sub_trainx:",trainX.shape, trainX)
    print("sub_testx", testX.shape, testX)
    print("sub_testy", testy.shape, testy)
        
    # zero-offset class values to perform one-hot encode (default values 1-6)
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    print("sub_trainy:",trainy.shape, trainy)
    print("sub_testy", testy.shape, testy)

    trainy = tf.keras.utils.to_categorical(trainy)

    testy = tf.keras.utils.to_categorical(testy)
    
    return  trainX, trainy, testX, testy



def load_uci_dataset(pathPrefix, numFeat):
    # load train dataset
    trainX, trainy = load_dataset_group("train", pathPrefix + "/", numFeat)

    print(trainX.shape, trainy.shape)

    # load test dataset
    testX, testy = load_dataset_group("test", pathPrefix + "/", numFeat)

    print(testX.shape, testy.shape)

   
    # zero-offset class values to perform one-hot encode (default values 1-6)
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y

    trainy = tf.keras.utils.to_categorical(trainy)

    testy = tf.keras.utils.to_categorical(testy)

    if sys.argv[1] == "bal":
        return balance_data(trainX, trainy, testX, testy)
    else:
        return trainX, trainy, testX, testy


def dataset_for_subject(Xtrain, ytrain, subjects_map, sub_id):
    # ottengo gli index delle righe corrispondenti al sub_id
    rows_indexes = [i for i in range(len(subjects_map)) if subjects_map[i]==sub_id]
    return Xtrain[rows_indexes, :], ytrain[rows_indexes]


def save_model(som, mod_path, typ, anova_val, som_dim):
    with open('./' + mod_path + '/anova_' + typ + '/' + anova_val + '/som' + som_dim + 'x' + som_dim + '.p', 'wb') as outfile:
        pickle.dump(som, outfile)

def save_dataset():
    return 