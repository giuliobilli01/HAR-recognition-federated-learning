import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from ML_utils import balance_data, feature_selection_anova
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from anovaf import get_anovaf, get_anovaF



def init_directories(w_path, plots_path, mod_path, np_arr_path, cent_type, fed_type, mean_path):
    if not os.path.exists("./" + plots_path):
        os.mkdir("./" + plots_path)
    if not os.path.exists("./" + plots_path + "/"+ cent_type):
        os.mkdir("./" + plots_path + "/"+ cent_type)
    if cent_type == "centr":
        
        if not os.path.exists("./" + plots_path + "/"+ cent_type + "/som_comp/"):
            os.mkdir("./" + plots_path + "/" + cent_type +  "/som_comp/")
    else:
        if not os.path.exists("./" + plots_path + "/"+ cent_type + "/" + fed_type):
            os.mkdir("./" + plots_path + "/" + cent_type + "/" + fed_type )
        if not os.path.exists("./" + plots_path + "/"+ cent_type + "/" + fed_type + "/som_comp/"):
            os.mkdir("./" + plots_path + "/" + cent_type + "/" + fed_type + "/som_comp/")

    
    if not os.path.exists("./" + mod_path):
        os.mkdir("./" + mod_path)
    if not os.path.exists("./" + mod_path + "/" + cent_type):
        os.mkdir("./" + mod_path + "/" + cent_type)
    if not cent_type == "centr":
        if not os.path.exists("./" + mod_path + "/" + cent_type + "/" + fed_type):
            os.mkdir("./" + mod_path + "/" + cent_type + "/" + fed_type)
  
    if not os.path.exists("./" + np_arr_path):
        os.mkdir("./" + np_arr_path)
    if not os.path.exists("./" + np_arr_path + "/" + cent_type):
        os.mkdir("./" + np_arr_path + "/" + cent_type)
    if not cent_type == "centr":
        if not os.path.exists("./" + np_arr_path + "/" + cent_type + "/" + fed_type):
            os.mkdir("./" + np_arr_path + "/" + cent_type + "/" + fed_type)

    if not os.path.exists("./" + mean_path):
        os.mkdir("./" + mean_path)
    if cent_type == "no-centr":
        if not os.path.exists("./" + mean_path + "/"+ cent_type + "/"):
            os.mkdir("./" + mean_path + "/"+ cent_type + "/")
        

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

    loaded = np.array(loaded[0][:, :numFeat])

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


def create_subjects_datasets(anova_selection):
    pathPrefix = "./UCI HAR Dataset/"
    sub_map_train = load_file("./UCI HAR Dataset/train/subject_train.txt")
    sub_map_test = load_file("./UCI HAR Dataset/test/subject_test.txt")

    sub_map = np.concatenate((sub_map_train, sub_map_test))

    train_subjects = np.unique(sub_map)
    print("train_subjects", train_subjects)

    # carico e unisco il dataset (considerando 265 feature)
    uci_x_train, uci_y_train = load_dataset_group("train", pathPrefix, 265)
    uci_x_test, uci_y_test = load_dataset_group("test", pathPrefix, 265)

    # selezione feature ANOVA
    a_y_train = uci_y_train - 1
    a_y_test = uci_y_test - 1
        
    var_avg_c, var_min_c  = get_anovaf(uci_x_train, tf.keras.utils.to_categorical(a_y_train), uci_x_test, tf.keras.utils.to_categorical(a_y_test))
    #var_avg_c, var_min_c  = get_anovaF(uci_x_train, tf.keras.utils.to_categorical(a_y_train), uci_x_test, tf.keras.utils.to_categorical(a_y_test))

    anova_x_train, anova_x_test = feature_selection_anova(uci_x_train, uci_x_test, 0.8, var_avg_c)

    anova_X = np.concatenate((anova_x_train, anova_x_test))

    print("anova_X", anova_X.shape)

    X = np.concatenate((uci_x_train, uci_x_test))
    y = np.concatenate((uci_y_train, uci_y_test))

    for subject in train_subjects:
        # prendo il dataset del soggetto corrispondente all'index
        datasetX = []
        datasety = []

        if anova_selection:
            datasetX, datasety = dataset_for_subject(anova_X, y, sub_map, subject)
        else:
            datasetX, datasety = dataset_for_subject(X, y, sub_map, subject)

        # split subject dataset in 70% train and 30% test
        s_trainX, s_testX, s_trainy, s_testy = train_test_split(datasetX, datasety, train_size=0.70, random_state=42, shuffle=True, stratify=datasety)

        groups = ["train", "test"]
        # salvo il dataset in file csv
        for group in groups:
            if not os.path.exists("./UCI HAR Dataset split/"+ group +"/subject-" + str(subject)):
                os.mkdir("./UCI HAR Dataset split/"+ group +"/subject-" + str(subject))

        save_dataset(s_trainX, s_trainy, "./UCI HAR Dataset split/" + "train/" + "subject-" + str(subject) + "/", subject, "train")
        save_dataset(s_testX, s_testy, "./UCI HAR Dataset split/" + "test/" + "subject-" + str(subject) + "/", subject, "test")

def load_subjects_group(group, subjects_to_ld, output_mode, pathPrefix=""):
    

    X_lst = list()
    y_lst = list()

    for sub in subjects_to_ld:
        pathX = pathPrefix + group + "/" + "subject-" + str(sub) + "/" + "X" + group + ".csv"
        pathy = pathPrefix + group + "/" + "subject-" + str(sub) + "/" + "y" + group + ".csv"

        s_X = load_file(pathX)
        s_y = load_file(pathy)

        print("s X", s_X.shape)
        print("s y", s_y.shape)

        X_lst.append(s_X)
        y_lst.append(s_y)
    
    if (output_mode == "concat"):
        X = np.concatenate(X_lst, axis=0)
        y = np.concatenate(y_lst, axis=0)

        print("X", X.shape)
        print("y", y.shape)

        return X, y
    else:
        return X_lst, y_lst

    
    
        

def load_subjects_dataset(subjects_to_ld, output_mode):
    pathPrefix = "./UCI HAR Dataset split/"

    trainX, trainy = load_subjects_group("train", subjects_to_ld, output_mode, pathPrefix)
    testX, testy = load_subjects_group("test", subjects_to_ld, output_mode, pathPrefix)

    # zero-offset class values to perform one-hot encode (default values 1-6)
    if output_mode == "concat":
        trainy = trainy - 1
        testy = testy - 1
        
        # one hot encode y
        trainy = tf.keras.utils.to_categorical(trainy)
    
        testy = tf.keras.utils.to_categorical(testy)
    else:
        for idx, elem in enumerate(trainy):
            trainy[idx] -= 1
            trainy[idx] = tf.keras.utils.to_categorical(trainy[idx])

        for idx, elem in enumerate(testy):
            testy[idx] -= 1
            testy[idx] = tf.keras.utils.to_categorical(testy[idx])
    
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


def save_model(som, mod_path, typ, anova_val, som_dim, centr_type, fed_type):
    with open('./' + mod_path + "/" + centr_type + "/" + fed_type  +  '/anova_' + typ + '/' + anova_val + '/som' + som_dim + 'x' + som_dim + '.pkl', 'wb') as outfile:
        pickle.dump(som, outfile)

def save_dataset(X, y, pathPrefix, sub_index, dataset_type):
    
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    # salvo il dataset del soggetto in un csv
    df_X.to_csv(pathPrefix + "X" + dataset_type+ ".csv" , sep=" ", index=False, header=False)
    df_y.to_csv(pathPrefix + "y"+ dataset_type+ ".csv", sep=" ", index=False, header=False)


def split_list(sub_list):
    chunk_list = []
    start = 0
    end = len(sub_list) 
    step = 3
    for i in range(start, end, step): 
        x = i 
        chunk_list.append(sub_list[x:x+step]) 
    return chunk_list

def calculate_class_distribution():
    pathPrefix = "./UCI HAR Dataset split/"

    subjects_to_ld=np.arange(1,31)
    print("subs to ld", subjects_to_ld)

    trainX, trainy = load_subjects_group("train", subjects_to_ld, "separated", pathPrefix)
    testX, testy = load_subjects_group("test", subjects_to_ld, "separated", pathPrefix)

    print("len testy", testy[0].shape)
    print("len trainy", trainy[0].shape)

    # per ogni soggetto devo contare quanti elementi ha per ogni classe
    subjects_dictionary = {}
    for sub_index in range(30):
        sub_string = "subject-" + str(sub_index)
        class_elements = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for index, elem in enumerate(trainy[sub_index]):
            class_elements[elem[0]] += 1
        for index, elem in enumerate(testy[sub_index]):
            class_elements[elem[0]] += 1
        subjects_dictionary.update({sub_string: class_elements})
    
    # creazione dati per plot
    subjects_keys = subjects_dictionary.keys()
    subjects = []
    for elem in subjects_keys:
        subjects.append(elem)

    splitted_subjects = split_list(subjects)
    print("splitted", splitted_subjects)

    for idx, subjects_chunck in enumerate(splitted_subjects):
        plot_class_distribution(subjects_chunck, subjects_dictionary, idx + 1)
    
def plot_class_distribution(subjects, subjects_dictionary, idx_group):
    class_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    for idx, key in enumerate(subjects):
        for c_num in range(len(class_dict)):
            class_dict[c_num + 1].append(subjects_dictionary[key][c_num + 1])
   
    # costruzione plot a barre
    x = np.arange(len(subjects))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for class_num, class_elems in class_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, class_elems, width, label=str(class_num))
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    print(subjects)
    ax.set_ylabel('Number of elements')
    ax.set_title(f'Class ditributions subs({(idx_group - 1)*3}-{idx_group * 3})')
    ax.set_xticks(x + width, subjects)
    ax.legend(loc='upper left', ncols=6)
    ax.set_ylim(0, 250)

    plt.savefig(f"./class-distribution_subs({(idx_group - 1)*3}-{idx_group * 3}).png")
        



        