import pandas as pd
import sys
import numpy as np
import tensorflow as tf
import os
import random
import flwr as fl
import ray
import json
import itertools

from logging import INFO, DEBUG
from flwr.common.logger import log
from typing import Dict, Tuple, List
from sklearn.metrics import classification_report
from minisom import MiniSom
from flwr.common import NDArrays, Scalar, Metrics
from utils import init_directories, load_subjects_dataset, create_subjects_datasets, calculate_class_distribution, onehot_decoding, filter_list_by_index
from plots import plot_fed_nofed_centr_comp, plot_som_comp_dim, plot_boxplot, plot_som, plot_cluster_comparison
from ML_utils import calculate_subjects_accs_mean, save_accuracies, save_federated_combination, filter_combinations, find_missing_element, get_saved_combinations_accs

# input parameter: 
#   1) gen / load: genera il dataset dei singoli soggetti o carica quello già creato
#   2) num: numero di soggetti di cui prendere il dataset default = 2
#   3) centr: eseguire train centralizzato oppure no
#   4) sing: eseguire train singolo oppure no
#   5) fed: eseguire train federated oppure no
#   6) y / n: salva i grafici e i vari dati generati oppure non salvarli
#   7) num: dimensione minima della som
#   8) num: dimensione massima della som

# accs strutture di supporto
plot_labels_lst = []
single_accs_combinations = {}
federated_accs_combinations = {}
centr_accs_combinations = {}


# default setup delle variabili di path e parametri
save_data = "y"
w_path = "weights UCI"
plots_path = "plots UCI"
mod_path = "som_models UCI"
np_arr_path = "np_arr UCI"
mean_path = "subjects_accs mean"
accs_path = "subjects_accs"
dataset_type = "split"
min_som_dim = 20
max_som_dim = 20
current_som_dim = min_som_dim
old_som_dim = 0
step = 10
exec_n = 1
total_execs = 0
actual_exec = 0
subjects_number = 2
centralized = False
single = False
federated = False

fed_Xtrain = []
fed_ytrain = []
fed_Xtest = []
fed_ytest = []

subjects_number = sys.argv[2] 

if sys.argv[3] == "centr":
    centralized = True

if sys.argv[4] == "sing":
    single = True

if sys.argv[5] == "fed":
    federated = True
    
if sys.argv[6] == 'n':
    save_data = "n"


if len(sys.argv) >= 8:
    min_som_dim = sys.argv[6]
    max_som_dim = sys.argv[7]


init_directories(w_path, plots_path, mod_path, np_arr_path, mean_path)


train_iter_lst = [100]  # , 250, 500, 750, 1000, 5000, 10000, 100000
dimensions = [15, 30]

divider = 10000
range_lst = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#range_lst = [8000]
    
total_execs = (
        (((max_som_dim + step) - min_som_dim) / step) * exec_n #* len(range_lst)
    )


#####FEDERATED FUNCTIONS

# Return the current local model parameters
def get_parameters(som):
    return [som.get_weights()]

def set_parameters(som, parameters):
    som._weights = parameters[0]

def classify_fed(som, data, X_train, y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap contiene una classificazione di campione in X_train 
    # con una delle classi in y (associazione neurone-label)
    
    new_y_train = list()
    for idx, item in enumerate(y_train):
        # inserisco in y gli index di ogni classe invertendo il one-hot encode
        new_y_train.append(np.argmax(y_train[idx]))

    winmap = som.labels_map(X_train , new_y_train)
    default_class = np.sum( list (winmap.values())).most_common()[0][0]
    
    result = []
    for d in data :
        win_position = som.winner( d )
        if win_position in winmap :
            result.append( winmap [ win_position ].most_common()[0][0])
        else :
            result.append( default_class )
    return result

class SomClient(fl.client.NumPyClient):
    def __init__(self, som, Xtrain, ytrain, Xtest, ytest , train_iter, cid):
        self.som = som
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.train_iter = train_iter
        self.Xtest = Xtest
        self.ytest = ytest
        self.cid = cid

    # Return the current local model parameters
    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.som)
    
    # Receive model parameters from the server, 
    # train the model parameters on the local data, 
    # and return the (updated) model parameters to the server
    def fit(self, parameters, config):
        set_parameters(self.som, parameters)
        self.som.train_random(self.Xtrain, self.train_iter, verbose=False)
        return get_parameters(self.som), len(self.Xtrain), {}
    
    # Receive model parameters from the server,
    # evaluate the model parameters on the local data, 
    # and return the evaluation result to the server
    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
        new_y_test = list()
        for idx, item in enumerate(self.ytest):
            # inserisco in new_test_y gli index di ogni classe invertendo il one-hot encode
            new_y_test.append(np.argmax(self.ytest[idx]))
        set_parameters(self.som, parameters)
        class_report = classification_report(
            new_y_test,
            classify_fed(
                self.som,
                self.Xtest,
                self.Xtrain,
                self.ytrain
            ),
            zero_division=0.0,
            output_dict=True,
        )

        return float(0), len(self.Xtest), {"accuracy": float(class_report["accuracy"]), "weights": get_parameters(self.som)[0]}


def client_fn(cid) -> SomClient:
    neurons = current_som_dim
    train_iter = train_iter_lst[0]
    # prendo il dataset corrispondente al cid(client id)
    Xtrain = fed_Xtrain[int(cid)]
    ytrain = fed_ytrain[int(cid)]
    Xtest = fed_Xtest[int(cid)]
    ytest = fed_ytest[int(cid)]

    som = MiniSom(
            neurons,
            neurons,
            Xtrain.shape[1],
            sigma=5,
            learning_rate=0.1,
            neighborhood_function="gaussian",
            activation_distance="manhattan",
        )

    return SomClient(som, Xtrain, ytrain, Xtest, ytest, train_iter, cid)


def weighted_simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    w_accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    s_accuracies = [m["accuracy"] for _, m in metrics]
    clients_num = len(metrics)
    # Aggregate and return custom metric (weighted average)
    return {"w_accuracy": sum(w_accuracies) / sum(examples), "s_accuracy": sum(s_accuracies)/clients_num}

def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    
    s_accuracies = [m["accuracy"] for _, m in metrics]
    weights = [m["weights"] for _, m in metrics]
    clients_num = len(metrics)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(s_accuracies)/clients_num, "weights": weights[0]}
######

def append_accuracies(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    s_accuracies = [m["accuracy"] for _, m in metrics]
    weights = [m["weights"] for _, m in metrics]
    
    return {"accuracy": s_accuracies, "weights": weights[0]}

def train_federated(num_rounds, num_clients, loocv_type):
    #definiamo come strategia FedAvg che 
    # FedAvg takes the 100 model updates and, as the name suggests, 
    # averages them. To be more precise, it takes the weighted average 
    # of the model updates, weighted by the number of examples each client used for training.
    #  The weighting is important to make sure that each data example has
    #  the same “influence” on the resulting global model. If one client 
    # has 10 examples, and another client has 100 examples,
    #  then - without weighting - each of the 10 examples would
    #  influence the global model ten times as much as each of the 100 examples.
    strategy = None

    if loocv_type == "noloocv":
        strategy = fl.server.strategy.FedAvg(
           fraction_fit=1.0,
           fraction_evaluate=1.0,
           min_fit_clients=int(num_clients),
           min_evaluate_clients=int(num_clients),
           min_available_clients=int(num_clients),
           evaluate_metrics_aggregation_fn=append_accuracies,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
           fraction_fit=1.0,
           fraction_evaluate=1.0,
           min_fit_clients=int(num_clients),
           min_evaluate_clients=int(num_clients),
           min_available_clients=int(num_clients),
           evaluate_metrics_aggregation_fn=simple_average,
        )
    
    client_resources = {"num_cpus": 1, "num_gpus":0.5}
    hist = fl.simulation.start_simulation(
       client_fn = client_fn,
       num_clients = int(num_clients),
       config = fl.server.ServerConfig(num_rounds=num_rounds),
       strategy = strategy,
       client_resources = client_resources,
   )
   #    ray_init_args = {"num_cpus": 2, "num_gpus":0.0}
    return hist.metrics_distributed["accuracy"], hist.metrics_distributed["weights"]


def train_combinations_single(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld, dimension, run_type, loocv_type):

    global single_accs_combinations
    # devo ottenere tutte le possibili combinazioni di coppie di soggetti
    # conviene lavorare su un array di indici e poi prendere il test e il train corrispondente dalle liste.
    combinations = list(itertools.combinations(subjects_to_ld, 2))
    single_accs_combinations[loocv_type].setdefault(dimension, [])
    # per ogni combinazione devo eseguire il train sul secondo e il test sul primo
    for combination in combinations:
        trainX = trainX_lst[combination[1] - 1]
        trainy = trainy_lst[combination[1] - 1]
        testX = testX_lst[combination[0] - 1]
        testy = testy_lst[combination[0] - 1]

        accuracy = run_training_single(trainX, trainy, testX, testy, dimension, run_type)
        single_accs_combinations[loocv_type][dimension].append(accuracy)
                

def run_training_single(trainX, trainy, testX, testy, neurons, run_type):
    # eseguiamo il train e testiamo l'accuracy sul test
    n_neurons = m_neurons = neurons
    new_y_test = onehot_decoding(testy)

    som = MiniSom(
        n_neurons,
        m_neurons,
        trainX.shape[1],
        sigma=5,
        learning_rate=0.1,
        neighborhood_function="gaussian",
        activation_distance="manhattan",
    )
    som.random_weights_init(trainX)
    som.train_random(trainX, 100, verbose=False)

    class_report = classification_report(
       new_y_test,
       classify(
           som,
           testX,
           trainX,
           trainy,
           n_neurons,
           100,
           0,
           run_type,
       ),
       zero_division=0.0,
       output_dict=True,
    )

    return class_report["accuracy"]


def train_combinations_federated(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld, dimension, combinations, loocv_type):

    global federated_accs_combinations
    global fed_Xtrain
    global fed_ytrain
    global fed_Xtest
    global fed_ytest

    print("dim", dimension)
    filtered_combinations = filter_combinations(combinations, dimension)
    federated_accs_combinations[loocv_type].setdefault(dimension, [])
    print("subjects", subjects_to_ld)
    print("combinations", len(filtered_combinations))
    
    saved_accs = get_saved_combinations_accs(dimension)
    for accuracy in saved_accs:
        federated_accs_combinations[loocv_type][dimension].append(accuracy)

    for combination in filtered_combinations:
    
        test_subject = find_missing_element(combination, 1, 31)
        fed_Xtrain = filter_list_by_index(trainX_lst, combination)
        fed_ytrain = filter_list_by_index(trainy_lst, combination)
        fed_Xtest = filter_list_by_index(testX_lst, combination)
        fed_ytest = filter_list_by_index(testy_lst, combination)
    
        loocv_Xtrain = trainX_lst[test_subject[0] - 1]
        loocv_ytrain = trainy_lst[test_subject[0] - 1]
        loocv_Xtest = testX_lst[test_subject[0] - 1]
        loocv_ytest = testy_lst[test_subject[0] - 1]
        _, weights = train_federated(5, len(fed_Xtrain))
        accuracy = test_combination_federated(loocv_Xtrain, loocv_ytrain, loocv_Xtest, loocv_ytest, dimension, weights)
        save_federated_combination(test_subject, accuracy, dimension)
        federated_accs_combinations[loocv_type][dimension].append(accuracy)


def test_combination_federated(loocv_Xtrain, loocv_ytrain, loocv_Xtest, loocv_ytest, dim, weights ):
     # una volta presi i pesi devo creare un instanza di una som
    # in cui caricare i pesi ed eseguire il test
    som = MiniSom(
            dim,
            dim,
            loocv_Xtrain.shape[1],
            sigma=5,
            learning_rate=0.1,
            neighborhood_function="gaussian",
            activation_distance="manhattan",)
    som._weights = weights[-1][1]

    new_y_test = onehot_decoding(loocv_ytest)
    class_report = classification_report(
        new_y_test,
        classify_fed(
            som,
            loocv_Xtest,
            loocv_Xtrain,
            loocv_ytrain,
        ),
        zero_division=0.0,
        output_dict=True,
    )

    return class_report["accuracy"]    

def train_combinations_centralized(dimension, combinations, loocv_type):

    centr_accs_combinations[loocv_type].setdefault(dimension, [])
    for combination in combinations:
        trainX, trainy, testX, testy = load_subjects_dataset(combination, "concat")
        print("trainx", trainX.shape)
        test_subject = find_missing_element(combination, 1, 31)
        loocv_Xtrain, loocv_ytrain, loocv_Xtest, loocv_ytest = load_subjects_dataset(test_subject, "concat")

        # eseguire il train sul dataset concatenato
        accuracy = run_training_centr(trainX, trainy, loocv_Xtest, loocv_ytest, dimension)
        centr_accs_combinations[loocv_type][dimension].append(accuracy)
        


def run_training_centr(trainX, trainy, testX, testy, neurons):
    # eseguiamo il train e testiamo l'accuracy sul test
    n_neurons = m_neurons = neurons
    new_y_test = onehot_decoding(testy)

    som = MiniSom(
        n_neurons,
        m_neurons,
        trainX.shape[1],
        sigma=5,
        learning_rate=0.1,
        neighborhood_function="gaussian",
        activation_distance="manhattan",
    )
    som.random_weights_init(trainX)
    som.train_random(trainX, 100, verbose=False)

    class_report = classification_report(new_y_test, classify_fed(som, testX, trainX, trainy),  zero_division=0.0, output_dict=True)
    return class_report["accuracy"]




def classify(som, data, X_train, y_train, neurons, train_iter, subj, run_type):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap contiene una classificazione di campione in X_train 
    # con una delle classi in y (associazione neurone-label)

    centr_type = "centr"
    fed_type = "no-fed"
    y = onehot_decoding(y_train)
    
    if not run_type == "centr":
        centr_type = "no-centr"
    
        if run_type == "no-fed":
            fed_type = "no-fed"

    winmap = som.labels_map(X_train , y)
    default_class = np.sum( list (winmap.values())).most_common()[0][0]

    if save_data == 'y':
        final_map = {}

        for idx, val in enumerate(winmap):
            final_map.update({(val[0] * neurons) + val[1]: winmap[val].most_common()[0][0]})

        final_map_lst = []
        pos_count = 0
        w_tot = pow(neurons, 2)
        for i in range(w_tot):
            if i not in final_map:
                final_map.update({i: default_class})

        # inserisce l'associazione neurone label all'interno di
        # final_map_lst in ordine, in modo da far coincidere l'index di ogni classe
        # con il neurone(codificato attraverso la formula (val[0] * neurons) + val[1])
        while len(final_map_lst) < len(final_map):
            for idx, val in enumerate(final_map):
                if int(val) == pos_count:
                    final_map_lst.append(final_map[val])
                    pos_count += 1

        final_map_lst = np.array(final_map_lst)
        if not centralized:
                if not os.path.exists(
                "./" + np_arr_path +"/" + centr_type + "/" + fed_type + "/"+ "subject-" + str(subj) + "/"
            ):
                    os.mkdir(
                        "./"
                        + np_arr_path
                        +"/" + centr_type + "/" + fed_type
                        + "/subject-" + str(subj)
                        + "/"
                    )   
        if not os.path.exists('./' + np_arr_path + "/" + centr_type + "/" + fed_type  + '/' + ( "subject-" + str(subj) + "/" if not centralized else "") ):
                os.mkdir('./' + np_arr_path + "/" + centr_type + "/" + fed_type + '/' + ( "subject-" + str(subj) + "/" if not centralized else "") )
        
        np.savetxt('./' + np_arr_path + "/" + centr_type + "/" + fed_type + '/' + ( "subject-" + str(subj) + "/" if not centralized else "")  + '/map_lst_iter-' + str(train_iter) + '_' + "subjects-" + str(subjects_number) + "_" + 
                "avg" + '_' + str(neurons) + '.txt', final_map_lst, delimiter=' ')

    result = []
    for d in data :
        win_position = som.winner( d )
        if win_position in winmap :
            result.append( winmap [ win_position ].most_common()[0][0])
        else :
            result.append( default_class )
    return result




def train_loocv_combinations(subjects_to_ld, dim, combinations, loocv_type):
    global accs_subjects_nofed
    global accs_subjects_fed
    global accs_subjects_centr

    trainX_lst, trainy_lst, testX_lst, testy_lst = []

    single_accs_combinations.setdefault(loocv_type, {})
    centr_accs_combinations.setdefault(loocv_type, {})
    federated_accs_combinations.setdefault(loocv_type, {})
    
    if loocv_type == "std-loocv":
        trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated", "full")
    elif loocv_type == "feat-loocv":
        
        if dim == 10:
            trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated", "full")
        elif dim == 15:
            trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated", "158")
        elif dim == 20:
            trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated", "89")
        elif dim == 30:
            trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated", "39")

    print("trainX lst shape", trainX_lst[0].shape)
    if federated:
        train_combinations_federated(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld, dim, combinations, loocv_type)
           
    if single:
        train_combinations_single(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld, dim, "no-centr", loocv_type)
    if centralized:
        train_combinations_centralized(dim, combinations, loocv_type)


def train_single(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld, dimension):
    global single_accs_combinations
   
    single_accs_combinations["noloocv"].setdefault(dimension, [])
    
    for subj_idx, subj in enumerate(subjects_to_ld):
        accuracy = run_training_single(trainX_lst[subj_idx], trainy_lst[subj_idx], testX_lst[subj_idx], testy_lst[subj_idx], dimension, "single")
        single_accs_combinations["noloocv"][dimension].append(accuracy)

       
def train_noloocv_centr(trainX, neurons):
    n_neurons = m_neurons = neurons

    som = MiniSom(
        n_neurons,
        m_neurons,
        trainX.shape[1],
        sigma=5,
        learning_rate=0.1,
        neighborhood_function="gaussian",
        activation_distance="manhattan",
    )
    som.random_weights_init(trainX)
    som.train_random(trainX, 100, verbose=False)

    return som    


def train_noloocv(subjects_to_ld, dim):

    global accs_subjects_nofed
    global accs_subjects_fed
    global accs_subjects_centr
    global fed_Xtrain
    global fed_ytrain
    global fed_Xtest
    global fed_ytest
    
    single_accs_combinations.setdefault("noloocv", {})
    centr_accs_combinations.setdefault("noloocv", {})
    federated_accs_combinations.setdefault("noloocv", {})

    trainX, trainy, testX, testy = load_subjects_dataset(subjects_to_ld, "concat", "full")
    trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated", "full")

    # deve fare il train e test per ogni soggetto
    if single:
        train_single(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld, dim)
    # deve fare il federated con train su tutti e 30 e test su tutti e 30
    if federated:
        fed_Xtrain = trainX_lst
        fed_ytrain = trainy_lst
        fed_Xtest = testX_lst
        fed_ytest = testy_lst
        federated_accs_combinations["noloocv"].setdefault(dim, [])

        accuracies, _ = train_federated(5, len(fed_Xtrain), "noloocv")
        print("fed accs", accuracies)
        federated_accs_combinations["noloocv"][dim] = accuracies[-1]

    if centralized:
    # deve fare il centralizzato con tutti e 30 e il test su ognuno
        centr_accs_combinations["noloocv"].setdefault(dim, [])
        trained_som = train_noloocv_centr(trainX, dim)
        # sulla som trainata devo fare la classificazione su tutti i test
        for subj_idx, subj in enumerate(subjects_to_ld):
            class_report = classification_report(onehot_decoding(testy_lst[subj_idx]), classify_fed(trained_som, testX_lst[subj_idx], trainX, trainy),  zero_division=0.0, output_dict=True)
            centr_accs_combinations["noloocv"][dim].append(class_report["accuracy"])
        

def run():
    global actual_exec
    global current_som_dim
    global fed_Xtrain
    global fed_ytrain
    global fed_Xtest
    global fed_ytest
    global accs_subjects_nofed
    global accs_subjects_fed
    global accs_subjects_centr

    # Use np.concatenate() Function
    
    if sys.argv[1] == "gen":
        create_subjects_datasets(False, 265)
    elif sys.argv[1] == "a-gen":
        create_subjects_datasets(True, 39)

    subjects_to_ld=random.sample(range(1, 31), int(subjects_number))
    subjects_to_ld.sort()
    print("subjects", subjects_to_ld)
    #calculate_class_distribution()

   
    # calcolo tutte le combinazioni possibili in cui ci sono 29 elementi
    # e uno rimane fuori
    combinations = list(itertools.combinations(subjects_to_ld, len(subjects_to_ld) - 1))


    for dim in dimensions:
        current_som_dim = dim
        train_noloocv(subjects_to_ld, dim)
        train_loocv_combinations(subjects_to_ld, dim, combinations, "std-loocv")
        train_loocv_combinations(subjects_to_ld, dim, combinations, "feat-loocv")


    save_accuracies(single_accs_combinations, federated_accs_combinations, centr_accs_combinations, accs_path, dimensions)
    plot_boxplot(dimensions, accs_path)
    



if __name__ == "__main__":
    run()
