import os
import re
import random
import time
import numpy as np
import pandas as pd
import collections
from pathlib import Path
from utils.path_funcs import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import model_training.patch_model_settings 


def isEven(n):
    return n % 2 == 0

def create_diamond_shape_using_powers_of_two(starting_num_neurons,n):
    result = []
    if isEven(n=n):
        first_half = [starting_num_neurons*2**i for i in range(n)]
        second_half = [starting_num_neurons*2**i for i in range(n, -1, -1)]
        result =  first_half + second_half
    else:
        first_half = [starting_num_neurons*2**i for i in range(n+1)]
        second_half = [starting_num_neurons*2**i for i in range(n-1, -1, -1)]
        result =  first_half + second_half
    return result

def print_training_results():
    path_to_patch_model_training_data = get_patch_model_training_data_folder_path()
    files = [i for i in os.listdir(path=path_to_patch_model_training_data) if "test" not in i and "rand" in i]
    print(files)
    print(len(files))
    for i in files:
        current_file_path = os.path.join(path_to_patch_model_training_data, i)
        df = pd.read_csv(current_file_path)
        print(i)
        print(df.shape)
        # print(df.columns)
        print(df.iloc[-1])
        # print(df.describe())
        print("-"*10)
        print( )

def get_indices_of_largest_N_numbers_in_a_list(list_):
    result = {}
    for i in range(len(list_)):
        result[i] = list_[i]
    result = sorted(result.items(),key=lambda x:x[1], reverse=True)
    return result


def get_top_N_largest_nums_indices_in_list(list_, N=5):
    result = {}
    count = 0
    for key, value in get_indices_of_largest_N_numbers_in_a_list(list_):
        result[key] = value
        count += 1
        if count == N:
            return result
            
    return result


def plot_training_history(model_name,func='loss',plot_name_and_abs_path='',add_to_title='' ,
                          ext='jpg', show=True, y_range=(1e-4,2e+0)):
    path_to_csv = get_patch_model_training_data_file_abs_path_by_model_name(model_name)
    df = pd.read_csv(path_to_csv)
    title = f"{func} over {len(df.index)} epochs"
    # print("loaded_model.history: ", loaded_model.history)
    # print("loaded_model.history.keys(): ", loaded_model.history.keys())
    # print("loaded_model.history.head(): ", loaded_model.history.head())
    data1 = df[func].tolist()
    data2 = df[f"val_{func}"].tolist()
    # print("data1: ", data1)
    lowest_point1 = (len(data1)-1, data1[-1])
    lowest_point2 = (len(data2)-1, data2[-1])


   
    if func != 'loss':
        title = f"Accuracy over {len(df.index)} epochs"
    
    if add_to_title != '':
        title = title + "\n" + add_to_title

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_range[0],top=y_range[1])
    plt.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))

    text1 = str(lowest_point1[1])
    text2 = str(lowest_point2[1])
    ax.plot(data1, color='red', label=func+f'-(final = {text1})')
    ax.plot(data2, color='blue', label=f'val_{func}'+f'-(final = {text2})')

    # final_text = AnchoredText(,loc)

    # ax.annotate(text1,
    #         xy=lowest_point1, xycoords='axes fraction',
    #         xytext=(-1.5,-1.5), textcoords='offset points',
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # ax.annotate(text2,
    #         xy=lowest_point2, xycoords='axes fraction',
    #         xytext=(-1.5,1.5), textcoords='offset points',
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    ax.legend()
    ax.grid(True,which="both") 
    ax.set_title(title)
    ax.set_xlabel("epochs")
    # ax.set_xticks(np.arange(0,100+1, 5))
    ax.set_ylabel(func)

    if plot_name_and_abs_path:
        print(f"plot_name_and_abs_path: {plot_name_and_abs_path}")
        plt.savefig(plot_name_and_abs_path)
    if show:
        plt.show()


def get_batch_size_from_name(name):
    pattern = r"bs\-([0-9]*)"
    batch_size = re.search(pattern=pattern, string=name)
    return batch_size.group(1)

def get_shape_from_name(name):
    pattern = r"shape(\-([0-9]*))*"

    shape = re.search(pattern=pattern, string=str(name))
    shape = shape.group(0).replace("shape", "")
    shape = shape[1:]
    return shape

def re_enumerate_epochs_in_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col=False)
    # print(f"df before:\n{df}")
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
    df = df.drop('epoch', axis=1)
    df = df.reset_index().rename(columns={'index': 'epoch'})
    # print(f"df after:\n{df}")
    df.to_csv(csv_file_path, index=False)


def bar_plot_patch_model_performance_for_all_patches(patch_model_name, data, plot_name=None):
    t1 = time.time()
    X_train, X_test, Y_train, Y_test = data
    # patch_model_abs_path = get_abs_saved_patch_models_folder_path_with_model_name(patch_model_name)
    patch_model = model_training.patch_model_settings.PatchClassificationModel(name=patch_model_name)
    predictions = patch_model.predict(X_test)
    patches = [i for i in range(96)]
    Y_train = Y_train.numpy().tolist()
    # print("Y_train = ", Y_train)
    # print("Y_train.numpy().tolist() = ", Y_train.numpy().tolist())
    training_samples_per_patch = dict(sorted(collections.Counter(Y_train).items()))
    print(f"training_samples_per_patch: ", training_samples_per_patch)
    correct_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] == predictions[i]]).items())).values())
    incorrect_predictions = list(dict(sorted(collections.Counter([predictions[i] for i in range(len(predictions)) if Y_test[i] != predictions[i]]).items())).values())
    print(f"correct_predictions: {correct_predictions}")
    print(f"incorrect_predictions: {incorrect_predictions}")
    predictions = {
        "correctly predicted": correct_predictions,
        "incorrectly predicted": incorrect_predictions
    }
    width=1.5
    fig, ax = plt.subplots(figsize=(20, 10))
    factor = 4
    ax.bar([factor*i for i in training_samples_per_patch.keys()], training_samples_per_patch.values(), color="orange", width=width, label="Training points")
    ax.bar([(factor*i + width) for i in patches], predictions["correctly predicted"], width=width, label="Correctly predicted", color="green")
    ax.bar([(factor*i + width) for i in patches], predictions["incorrectly predicted"], width=width, label="Incorrectly predicted", bottom=predictions["correctly predicted"], color="red")

    ax.set_title("Number of points used for training compared to numbers of correct and incorrect predictions on validation data")
    ax.set_xticks([factor*i for i in training_samples_per_patch.keys()])
    
    ax.set_xticklabels(([str(i) for i in patches]), rotation=90)
    ax.legend(loc="upper right")
    print(f"total time in seconds: {time.time() - t1}")
    print(f"name: {plot_name}")
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()
    
