import numpy as np
import sys
import json

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

def feature_selection_anova(X_train, a_val, varianza_media_classi, divider):
    less_than_anova_vals = []
    greater_than_anova_vals = []
    # si sceglie l'index delle feature che andranno a comporre l'input del modello
    for idx, val in enumerate(varianza_media_classi):
        if val > a_val / divider:
            greater_than_anova_vals.append(idx)
        else:
            less_than_anova_vals.append(idx)
    
    return X_train[:, less_than_anova_vals]


def calculate_subjects_accs_mean(nofed_accs, min_som_dim, max_som_dm, step, mean_path, cent_type, fed_type):
    mean_dict = {}
    subjects_num = len(nofed_accs.keys())
    for dim in range(min_som_dim, max_som_dm + step, step):
        accumulatore = 0
        for subj_num in nofed_accs.keys():
            print("acc:", nofed_accs[subj_num][dim])
            accumulatore += nofed_accs[subj_num][dim]
    
        mean_dict.update({dim: accumulatore/subjects_num})
    
    #salvo il dizionario
    with open("./" + mean_path + "/" + cent_type + "/" + fed_type + "/" + "mean.txt", "w") as fp:
        json.dump(mean_dict, fp)


