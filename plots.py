import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import numpy as np
import os


def plot_som_comp(
    train_iter,
    accs_avg_mean,
    accs_avg_max,
    accs_avg_min,
    plot_labels_lst,
    save_data,
    centr_type,
    fed_type,
    subjects,
    plots_path,
    range_lst,
    divider,
    exec_n,
    subj,
    centralized,
    acc_mean_km=None,
    acc_min_km=None,
    acc_max_km=None,
):
    if acc_min_km is None:
        acc_min_km = {}
    if acc_max_km is None:
        acc_max_km = {}
    if acc_mean_km is None:
        acc_mean_km = {}
    name = "som"

    min_neurons = None
    plt.figure()

    if not os.path.exists("./" + plots_path +"/" + centr_type + "/" + fed_type + ("/subject-" + subj  if not centralized else "") + "/som_comp"+ "/"):
        os.mkdir("./" + plots_path +"/" + centr_type + "/" + fed_type + ("/subject-" + subj  if not centralized else "") + "/som_comp"+ "/")
    key_lst_km = []
    # k sono le dimensioni della som
    for k in accs_avg_mean.keys():
        keys_lst = []
        vals_lst = []
        # val sono i valori anova testati
        for val in accs_avg_mean[k].keys():
            keys_lst.append(str(val))
        for val in accs_avg_mean[k].values():
            vals_lst.append(val)
        plt.plot(keys_lst, vals_lst, label=str(k) + "x" + str(k), marker="o")
        key_lst_km = keys_lst
        # plt.xticks(np.array(anova_val_tested_global))
    # plt.xticks(anova_val_tested_global[0])
    plt.xlabel("Anova Threshold")
    plt.ylabel("Accuracy")
    string = (
        "Accuracies comparison choosing the mean of the variances per class"
    )
    plt.title(string)
    plt.legend()
    
    min_neurons = plot_labels_lst[0].split("x")[0]
    max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split("x")[0]
    step_neurons = 0
    if len(plot_labels_lst) > 1:
        step_val = plot_labels_lst[1].split("x")[0]
        step_neurons = int(step_val) - int(min_neurons)
    if save_data == "y":
        plt.savefig(
            "./"
            + plots_path
            +"/" + centr_type + "/" + fed_type
            + ( "/subject-" + subj  if not centralized else "")
            + "/som_comp"
            + "/"
            + name
            + "_comp_avg_mean_iter-"
            + str(train_iter)
            + "_subjects-" + str(subjects)
            + "_range("
            + str(range_lst[0] / divider)
            + ","
            + str(range_lst[len(range_lst) - 1] / divider)
            + ")_minneur-"
            + str(min_neurons)
            + "maxneur-"
            + str(max_neurons)
            + "_step-"
            + str(step_neurons)
            + "_execs-"
            + str(exec_n)
            + ".png"
        )
    plt.close()
    plt.figure()
    # print(anova_val_tested_global)
    for k in accs_avg_max.keys():
        keys_lst = []
        vals_lst = []
        for val in accs_avg_max[k].keys():
            keys_lst.append(str(val))
        for val in accs_avg_max[k].values():
            vals_lst.append(val)
        plt.plot(keys_lst, vals_lst, label=str(k) + "x" + str(k), marker="o")
    # plt.xticks(anova_val_tested_global[0])
    plt.xlabel("Anova Threshold")
    plt.ylabel("Accuracy")
    string = (
        "Accuracies comparison choosing the mean of the variances per class per f."
    )
    # plt.title(string)
    plt.legend()
    # plt.show()
    # step_val = 0
    min_neurons = plot_labels_lst[0].split("x")[0]
    max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split("x")[0]
    step_neurons = 0
    if len(plot_labels_lst) > 1:
        step_val = plot_labels_lst[1].split("x")[0]
        step_neurons = int(step_val) - int(min_neurons)
    if save_data == "y":
        plt.savefig(
            "./"
            + plots_path
            +"/" + centr_type + "/" + fed_type
            + ( "/subject-" + subj if not centralized else "")
            + "/som_comp"
            + "/"
            + name
            + "_comp_avg_max_iter-"
            + str(train_iter)
            + "_subjects-" + str(subjects) 
            + "_range("
            + str(range_lst[0] / divider)
            + ","
            + str(range_lst[len(range_lst) - 1] / divider)
            + ")_minneur-"
            + str(min_neurons)
            + "-maxneur"
            + str(max_neurons)
            + "_step-"
            + str(step_neurons)
            + "_execs-"
            + str(exec_n)
            + ".png"
        )
    plt.close()
    plt.figure()
    # print(anova_val_tested_global)
    for k in accs_avg_min.keys():
        keys_lst = []
        vals_lst = []
        for val in accs_avg_min[k].keys():
            keys_lst.append(str(val))
        for val in accs_avg_min[k].values():
            vals_lst.append(val)
        plt.plot(keys_lst, vals_lst, label=str(k) + "x" + str(k), marker="o")
    # plt.xticks(anova_val_tested_global[0])
    plt.xlabel("Anova Threshold")
    plt.ylabel("Accuracy")
    string = (
        "Accuracies comparison choosing the mean of the variances per class per f."
    )
    # plt.title(string)
    plt.legend()
    # plt.show()
    # step_val = 0
    min_neurons = plot_labels_lst[0].split("x")[0]
    max_neurons = plot_labels_lst[len(plot_labels_lst) - 1].split("x")[0]
    step_neurons = 0
    if len(plot_labels_lst) > 1:
        step_val = plot_labels_lst[1].split("x")[0]
        step_neurons = int(step_val) - int(min_neurons)
    if save_data == "y":
        plt.savefig(
            "./"
            + plots_path
            +"/" + centr_type + "/" + fed_type
            + ( "/subject-" + subj if not centralized else "")
            + "/som_comp"
            + "/"
            + name
            + "_comp_avg_min_iter-"
            + str(train_iter)
            + "_subjects-" + str(subjects) 
            + "_range("
            + str(range_lst[0] / divider)
            + ","
            + str(range_lst[len(range_lst) - 1] / divider)
            + ")_minneur-"
            + str(min_neurons)
            + "maxneur-"
            + str(max_neurons)
            + "_step-"
            + str(step_neurons)
            + "_execs-"
            + str(exec_n)
            + ".png"
        )
    plt.close()
   


def plot_som(som, X_train, y_train, path, a_val, n_feat, save_data, subjects, subj, centralized):
    plt.figure(figsize=(9, 9))

    plt.pcolor(
        som.distance_map(scaling="mean").T, cmap="bone_r"
    )  # plotting the distance map as background
    plt.colorbar()

    # Plotting the response for each pattern in the iris dataset
    # different colors and markers for each label
    markers = ["o", "s", "D", "v", "1", "P"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    activity = ["walking", "w. upst", "w. downst", "sitting", "standing", "laying"]
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(
            w[0] + 0.5,
            w[1] + 0.5,
            markers[np.argmax(y_train[cnt])],
            markerfacecolor="None",
            markeredgecolor=colors[np.argmax(y_train[cnt])],
            markersize=6,
            markeredgewidth=2,
            label=activity[np.argmax(y_train[cnt])],
        )
    mrk1 = mlines.Line2D(
        [],
        [],
        markeredgecolor=colors[0],
        marker=markers[0],
        markerfacecolor="None",
        markeredgewidth=2,
        linestyle="None",
        markersize=6,
    )
    mrk2 = mlines.Line2D(
        [],
        [],
        markeredgecolor=colors[1],
        marker=markers[1],
        markerfacecolor="None",
        markeredgewidth=2,
        linestyle="None",
        markersize=6,
    )
    mrk3 = mlines.Line2D(
        [],
        [],
        markeredgecolor=colors[2],
        marker=markers[2],
        markerfacecolor="None",
        markeredgewidth=2,
        linestyle="None",
        markersize=6,
    )
    mrk4 = mlines.Line2D(
        [],
        [],
        markeredgecolor=colors[3],
        marker=markers[3],
        markerfacecolor="None",
        markeredgewidth=2,
        linestyle="None",
        markersize=6,
    )
    mrk5 = mlines.Line2D(
        [],
        [],
        markeredgecolor=colors[4],
        marker=markers[4],
        markerfacecolor="None",
        markeredgewidth=2,
        linestyle="None",
        markersize=6,
    )
    mrk6 = mlines.Line2D(
        [],
        [],
        markeredgecolor=colors[5],
        marker=markers[5],
        markerfacecolor="None",
        markeredgewidth=2,
        linestyle="None",
        markersize=6,
    )
    by_label = dict(zip(activity, [mrk1, mrk2, mrk3, mrk4, mrk5, mrk6]))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    # plt.legend()
    # plt.show()
    if centralized:    
        plt.savefig(path + "subjects-" + str(subjects) + "_" + str(a_val) + "_" + str(n_feat) + ".png")
    else:
        plt.savefig(path + "subject-" + str(subj) + "_" + str(a_val) + "_" + str(n_feat) + ".png")

    plt.close()
