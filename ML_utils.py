import numpy as np
import sys
import json
import os


def find_missing_element(elem_list, start, end):
    test_array = np.array(elem_list)
    return np.setdiff1d(np.arange(start, end), test_array)

# balance data
def balance_data(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    bal_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    X_train_bal = np.empty((0, X.shape[1]))
    X_test_bal = np.empty((0, X.shape[1]))
    y_train_bal = np.empty((0, y.shape[1]))
    y_test_bal = np.empty((0, y.shape[1]))
    print("X:", X.shape)
    print("y:", y.shape)
    for idx, item in enumerate(X):
        bal_val = 1356 
        if sys.argv[4] == 'split':
            bal_val = X_train.shape[0] / 6
        if bal_dict[int(np.argmax(y[idx]))] <= bal_val:
            X_train_bal = np.concatenate((X_train_bal, [item]))
            y_train_bal = np.concatenate((y_train_bal, [y[idx]]))
            bal_dict[int(np.argmax(y[idx]))] += 1
        else:
            X_test_bal = np.concatenate((X_test_bal, [item]))
            y_test_bal = np.concatenate((y_test_bal, [y[idx]]))

        print("\rBalancing data progress: " + str(idx + 1) + "/" + str(len(X)), end="")
    return X_train_bal, y_train_bal, X_test_bal, y_test_bal

def feature_selection_anova(X_train, X_test, a_val, varianza_media_classi):
    less_than_anova_vals = []
    greater_than_anova_vals = []
    # si sceglie l'index delle feature che andranno a comporre l'input del modello
    for idx, val in enumerate(varianza_media_classi):
        if val > a_val:
            greater_than_anova_vals.append(idx)
        else:
            less_than_anova_vals.append(idx)
    
    return X_train[:, less_than_anova_vals], X_test[:, less_than_anova_vals]

def feature_selection_with_max(X_train, X_test, varianza_media_classi, max_n):
    selected_feature_idx = []

    # creo una versione ordinata dei valori anova
    sorted_anova = sorted(varianza_media_classi)[:max_n]
    print("sorted", sorted_anova)
    print("sorted len", len(sorted_anova))
    
    for idx, val in enumerate(varianza_media_classi):
        if val in sorted_anova:
            selected_feature_idx.append(idx)
    
    return X_train[:, selected_feature_idx], X_test[:, selected_feature_idx]





def calculate_subjects_accs_mean(nofed_accs, fed_accs, centr_accs, min_som_dim, max_som_dm, step, mean_path, subjects_loaded, centralized, single, federated, execution_num):
    mean_dict = {}
    subjects_num = len(subjects_loaded)
    print("nofed accs", nofed_accs)
    print("fed accs", fed_accs)
    print("centr accs", centr_accs)


    if os.path.exists("./" + mean_path + "/" + "mean.txt"): 
        with open ("./" + mean_path + "/"  + "mean.txt") as js:
            data = json.load(js)
            mean_dict = data
    
    subs_string = "subjects["
    for sub in subjects_loaded:
        subs_string += ("-" + str(sub))
    subs_string+="]"
    
    mean_dict.setdefault(subs_string, { "subjects": subjects_loaded, "nofed_accs": {}, "fed_accs": {}, "centr_accs": {}})

    print("mean dict after set", mean_dict)
    
    for execution in range(execution_num):
        for dim in range(min_som_dim, max_som_dm + step, step):
            accumulatore = 0
            if single:
                for subj_num in nofed_accs.keys():
                    accumulatore += nofed_accs[subj_num][dim][execution]
                
                mean_dict[subs_string]["nofed_accs"].setdefault(str(dim), [])

                mean_dict[subs_string]["nofed_accs"][str(dim)].append(accumulatore/subjects_num)

            if federated:
                mean_dict[subs_string]["fed_accs"].setdefault(str(dim), [])
                mean_dict[subs_string]["fed_accs"][str(dim)].append(fed_accs[dim][execution]["accuracy"])

            if centralized:
                mean_dict[subs_string]["centr_accs"].setdefault(str(dim), [])
                mean_dict[subs_string]["centr_accs"][str(dim)].append(centr_accs[dim][execution])
    #salvo il dizionario
    with open("./" + mean_path + "/" + "mean.txt", "w") as fp:
        json.dump(mean_dict, fp, indent=4)


def save_accuracies(single_accs, federated_accs, centr_accs, accs_path, dimensions):
    accs_dict = {}
    if os.path.exists("./" + accs_path + "/" + "accs.txt"): 
        with open ("./" + accs_path + "/"  + "accs.txt") as js:
            data = json.load(js)
            accs_dict = data

    accs_dict.setdefault("noloocv", {})
    accs_dict.setdefault("std-loocv", {})
    accs_dict.setdefault("feat-loocv", {})

    for loocv_type in ["noloocv", "std-loocv", "feat-loocv"]:
        accs_dict[loocv_type].setdefault("single", {})
        accs_dict[loocv_type].setdefault("federated", {})
        accs_dict[loocv_type].setdefault("centralized", {})

    print("accs dict", accs_dict)

    for loocv_type in ["noloocv", "std-loocv", "feat-loocv"]:
        for dim in dimensions: 
            accs_dict[loocv_type]["single"].update({str(dim): single_accs[loocv_type][dim]})
            accs_dict[loocv_type]["federated"].update({str(dim): federated_accs[loocv_type][dim]})
            accs_dict[loocv_type]["centralized"].update({str(dim): centr_accs[loocv_type][dim]})


    with open("./" + accs_path + "/" + "accs.txt", "w") as fp:
        json.dump(accs_dict, fp, indent=4)


def save_federated_combination(test_sub, accuracy, dimension):
    combination_dict = {}
    if os.path.exists(f"./combinations/accs{dimension}.txt"): 
        with open (f"./combinations/accs{dimension}.txt") as js:
            data = json.load(js)
            combination_dict = data

    combination_dict.setdefault(str(test_sub[0]), 0)
    combination_dict[str(test_sub[0])] = accuracy

    print("comb dict", combination_dict)
    with open(f"./combinations/accs{dimension}.txt", "w") as fp:
        json.dump(combination_dict, fp, indent=4)


def filter_combinations(combinations, dimension):
    combination_dict = {}
    if os.path.exists(f"./combinations/accs{dimension}.txt"): 
        with open (f"./combinations/accs{dimension}.txt") as js:
            data = json.load(js)
            combination_dict = data
    
    print("before comb", len(combinations))
    tested_subjects_keys = combination_dict.keys()
    tested_subjects = []
    for sub in tested_subjects_keys:
        tested_subjects.append(int(sub))

    filtered_comb = []
    for index, combination in enumerate(combinations):
        test_subject = find_missing_element(combination, 1, 31)
        if not test_subject[0] in tested_subjects:
            filtered_comb.append(combination)    

    return filtered_comb
    

def get_saved_combinations_accs(dimension):
    combination_dict = {}
    if os.path.exists(f"./combinations/accs{dimension}.txt"): 
        with open (f"./combinations/accs{dimension}.txt") as js:
            data = json.load(js)
            combination_dict = data
    accuracies = combination_dict.values()

    return accuracies