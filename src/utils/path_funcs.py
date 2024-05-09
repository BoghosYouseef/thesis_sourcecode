import os
from pathlib import Path 

def get_relative_data_folder_path():
    return Path("data/csv_files")

def get_relative_saved_models_folder_path():
    return Path("src/model_training/saved_models")

def get_list_of_elements_in_dir(dir_abs_path):
    return os.listdir(dir_abs_path)

def get_abs_path(file_name): 
    return os.path.abspath(file_name)

def get_relative_saved_plots_folder_path():
    return Path('data/plots')

def get_relative_model_training_data_folder_path():
    return Path('data/csv_files/model_training_history')

def get_patch_model_training_data_folder_path():
    return get_abs_path(os.path.join(get_relative_model_training_data_folder_path(), "patch_model"))