import os
import pandas as pd
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