import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import numpy as np
import os
import json


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

    if not os.path.exists(
        "./"
        + plots_path
        + "/"
        + centr_type
        + "/"
        + fed_type
        + ("/subject-" + subj if not centralized else "")
        + "/som_comp"
        + "/"
    ):
        os.mkdir(
            "./"
            + plots_path
            + "/"
            + centr_type
            + "/"
            + fed_type
            + ("/subject-" + subj if not centralized else "")
            + "/som_comp"
            + "/"
        )
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
    string = "Accuracies comparison choosing the mean of the variances per class"
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
            + "/"
            + centr_type
            + "/"
            + fed_type
            + ("/subject-" + subj if not centralized else "")
            + "/som_comp"
            + "/"
            + name
            + "_comp_avg_mean_iter-"
            + str(train_iter)
            + "_subjects-"
            + str(subjects)
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
    string = "Accuracies comparison choosing the mean of the variances per class per f."
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
            + "/"
            + centr_type
            + "/"
            + fed_type
            + ("/subject-" + subj if not centralized else "")
            + "/som_comp"
            + "/"
            + name
            + "_comp_avg_max_iter-"
            + str(train_iter)
            + "_subjects-"
            + str(subjects)
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
    string = "Accuracies comparison choosing the mean of the variances per class per f."
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
            + "/"
            + centr_type
            + "/"
            + fed_type
            + ("/subject-" + subj if not centralized else "")
            + "/som_comp"
            + "/"
            + name
            + "_comp_avg_min_iter-"
            + str(train_iter)
            + "_subjects-"
            + str(subjects)
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


def plot_som_comp_dim(
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
    exec_n,
    subj,
    centralized,
):

    if not os.path.exists(
        "./"
        + plots_path
        + "/"
        + centr_type
        + "/"
        + fed_type
        + ("/subject-" + subj if not centralized else "")
        + "/som_comp_dim"
        + "/"
    ):
        os.mkdir(
            "./"
            + plots_path
            + "/"
            + centr_type
            + "/"
            + fed_type
            + ("/subject-" + subj if not centralized else "")
            + "/som_comp_dim"
            + "/"
        )
    plt.figure()

    dim_lst = accs_avg_mean.keys()
    accs_lst = accs_avg_mean.values()
   
    plt.plot(dim_lst, accs_lst, label="accuracies", marker="o")
    # plt.xticks(np.array(anova_val_tested_global))
    # plt.xticks(anova_val_tested_global[0])
    plt.xlabel("Som Dimensions")
    plt.ylabel("Accuracy")
    string = "Accuracies comparison between different dimensions"
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
            + "/"
            + centr_type
            + "/"
            + fed_type
            + ("/subject-" + subj if not centralized else "")
            + "/som_comp_dim"
            + "/"
            + "som_comp_dims_mean-"
            + str(train_iter)
            + "_subjects-"
            + str(subjects)
            + "_minneur-"
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
    




    


def plot_som(
    som, X_train, y_train, path, n_feat, save_data, subjects, subj, centralized
):
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
        plt.savefig(path + "subjects-" + str(subjects) + "_" + str(n_feat) + ".png")
    else:
        plt.savefig(path + "subject-" + str(subj) + "_" + str(n_feat) + ".png")

    plt.close()


def plot_fed_nofed_centr_comp(mean_path, min_som_dim, max_som_dim, step, centralized, single, federated):
    data_dict = {}
    if os.path.exists("./" + mean_path + "/" + "mean.txt"): 
        with open ("./" + mean_path + "/" + "mean.txt") as js:
            data = json.load(js)
            data_dict = data
    
    for dim in range(min_som_dim, max_som_dim + step, step):
        subjects_nums = []
        nofed_accs = []
        fed_accs = []
        centr_accs = []
        plt.figure()
        for key in data_dict.keys():
            subjects_nums.append(len(data_dict[key]["subjects"]))
            if single:
                nofed_accs.append(data_dict[key]["nofed_accs"][str(dim)][0])
            if federated:
                fed_accs.append(data_dict[key]["fed_accs"][str(dim)][0][-1][-1])
            if centralized:
                centr_accs.append(data_dict[key]["centr_accs"][str(dim)][0])
        
        plt.plot(subjects_nums, nofed_accs, label="no-federated", marker='o')
        plt.plot(subjects_nums, fed_accs, label="federated", marker='o')
        plt.plot(subjects_nums, centr_accs, label="centralized", marker='o')

        plt.xlabel(f"Subjects")
        plt.ylabel("Accuracy")
        plt.title("Confronto tra federated, non federated e centralizzato")

        plt.legend()
        plt.savefig(
            "./"
            + mean_path
            + "/"
            + "som-" + str(dim) + "_comp-fed-nofed"
            + ".png"
        )
        plt.close()
    
def plot_cluster_comparison(mean_path, min_som_dim, max_som_dim, step, centralized, single, federated, execution_num):
    data_dict = {}
    if os.path.exists("./" + mean_path + "/" + "mean.txt"): 
        with open ("./" + mean_path + "/" + "mean.txt") as js:
            data = json.load(js)
            data_dict = data
    
    for key in data_dict.keys():
        subs_string = "subjects["
        for sub in data_dict[key]["subjects"]:
            subs_string += ("-" + str(sub))
        subs_string+="]"

        for dim in range(min_som_dim, max_som_dim + step, step):
            nofed_accs = []
            fed_accs = []
            centr_accs = []
            executions = np.arange(1, len(data_dict[key]["nofed_accs"][str(dim)]) + 1)
            print("execs", executions)
            plt.figure()
            for execution in range(len(executions)):
                if single:
                    nofed_accs.append(data_dict[key]["nofed_accs"][str(dim)][execution])
                if federated:
                    fed_accs.append(data_dict[key]["fed_accs"][str(dim)][execution][-1][-1])
                if centralized:
                    centr_accs.append(data_dict[key]["centr_accs"][str(dim)][execution])
            

            plt.plot(executions, nofed_accs, label="no-federated", marker='o')
            plt.plot(executions, fed_accs, label="federated", marker='o')
            plt.plot(executions, centr_accs, label="centralized", marker='o')

            plt.xlabel(f"Executions")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracies soggetti {subs_string}")

            plt.legend()
            plt.savefig(
                "./"
                + subs_string + "_"
                + "som-" + str(dim) + "_comp-fed-nofed"
                + ".png"
            )
            plt.close()


def define_box_properties(plot_groups, color_codes, labels):
   
    for idx, plot in enumerate(plot_groups):
        print("dix", idx)
        plt.plot([], c=color_codes[idx], label=labels[idx])
        plt.legend()
        


def plot_boxplot(dimensions, accs_path):
    accuracies_dict = {}
    if os.path.exists("./" + accs_path + "/" + "accs.txt"): 
        with open ("./" + accs_path + "/" + "accs.txt") as js:
            data = json.load(js)
            accuracies_dict = data
    # raggruppo i vari dati per le diverse dimensioni
    for loocv_type in ["noloocv", "std-loocv","feat-loocv"]:
        ten_lst = []
        fifteen_lst = []
        twenty_lst = []
        thirty_lst = []
        for key in accuracies_dict[loocv_type].keys():
            ten_lst.append(accuracies_dict[loocv_type][key]["10"]) 
            fifteen_lst.append(accuracies_dict[loocv_type][key]["15"])
            twenty_lst.append(accuracies_dict[loocv_type][key]["20"] )
            thirty_lst.append(accuracies_dict[loocv_type][key]["30"])

        colors = ['red', 'lightblue', 'lightgreen', 'orange']

        data_groups = [ten_lst, fifteen_lst, twenty_lst, thirty_lst]

        labels_lst = ["Single", "Federated", "Centralized"]

        width = 1/len(labels_lst)

        xlocations  = [ x*((1+ len(data_groups))*width) for x in range(len(ten_lst)) ]

        symbol      = 'r+'
        ymin        = min ( [ val  for dg in data_groups  for data in dg for val in data ] )
        ymax        = max ( [ val  for dg in data_groups  for data in dg for val in data ])

        ax = plt.gca()
        ax.set_ylim(ymin,ymax)

        ax.grid(True, linestyle='dotted')
        ax.set_axisbelow(True)

        plt.ylabel('Accuracies')
        plt.title('title')

        space = len(data_groups)/2
        offset = len(data_groups)/2

        group_positions = []
        for num, dg in enumerate(data_groups):    
            _off = (0 - space + (0.5+num))
            print(_off)
            group_positions.append([x+_off*(width+0.01) for x in xlocations])

        for dg, pos, c in zip(data_groups, group_positions, colors):
            boxes = ax.boxplot(dg, 
                        sym=symbol,
                        labels=['']*len(labels_lst),
            #            labels=labels_list,
                        positions=pos, 
                        widths=width, 
                        boxprops=dict(facecolor=c),
            #             capprops=dict(color=c),
            #            whiskerprops=dict(color=c),
            #            flierprops=dict(color=c, markeredgecolor=c),                       
                        medianprops=dict(color='grey'),
            #           notch=False,  
            #           vert=True, 
            #           whis=1.5,
            #           bootstrap=None, 
            #           usermedians=None, 
            #           conf_intervals=None,
                        patch_artist=True,
                        )
        ax.set_xticks( xlocations )
        ax.set_xticklabels( labels_lst, rotation=0 )
        define_box_properties(data_groups, colors, ["10", "15", "20", "30"])

        plt.savefig("./" + accs_path + "/boxplot_" + loocv_type +".png")
        plt.close()

