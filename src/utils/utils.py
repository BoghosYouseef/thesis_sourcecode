import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from utils.path_funcs import get_patch_model_training_data_folder_path

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


def plot_training_history(loaded_model,func='loss',name='', save=False,add_to_title='' ,ext='jpg', show=True):
    batch_size = get_batch_size_from_name(loaded_model.name)
    title = f"{func} over 100 epochs"
    print("loaded_model.history: ", loaded_model.history)
    print("loaded_model.history.keys(): ", loaded_model.history.keys())
    print("loaded_model.history.head(): ", loaded_model.history.head())
    data1 = loaded_model.history[func].tolist()
    data2 = loaded_model.history[f"val_{func}"].tolist()
    # print("data1: ", data1)
    lowest_point1 = (len(data1)-1, data1[-1])
    lowest_point2 = (len(data2)-1, data2[-1])


   
    if func != 'loss':
        title = f"Accuracy over {100} epochs"
    
    if add_to_title != '':
        title = title + "\n" + add_to_title

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")
    # ax.set_ylim(bottom=1e-2,top=2e+0)
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
    ax.set_xticks(np.arange(0,100+1, 5))
    ax.set_ylabel(func)
    if show:
            plt.show()


def get_batch_size_from_name(name):
    batch_size = name.split("-")[0].replace(".csv","")
    batch_size = name.replace(".keras","")
    return batch_size
