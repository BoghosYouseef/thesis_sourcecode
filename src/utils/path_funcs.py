import os
from pathlib import Path
from utils.decorators import make_output_path_obj

# from importlib_resources import path 

def get_abs_path_of_package_root():
    current_dir = os.path.dirname(os.path.realpath(__file__)) # get current abs path to this file
    abs_path = os.path.split(os.path.split(current_dir)[0])[0] # go up twice to go up to thesis_sourcecode
    
    return abs_path
    
@make_output_path_obj
def get_abs_raw_data_folder_path():
    return os.path.join(get_abs_path_of_package_root(),"data/csv_files/raw")

@make_output_path_obj
def get_abs_saved_models_folder_path():
    return os.path.join(get_abs_path_of_package_root(),"src/model_training/saved_models")

@make_output_path_obj
def get_abs_saved_patch_models_folder_path_with_model_name(name):
    return os.path.join(get_abs_saved_models_folder_path(),name + ".keras")


def get_list_of_elements_in_dir(dir_abs_path):
    return [f for f in Path(dir_abs_path).iterdir() if f.is_file()]

@make_output_path_obj
def get_abs_path(file_name): 
    return os.path.abspath(file_name)

@make_output_path_obj
def get_abs_saved_plots_folder_path():
    return os.path.join(get_abs_path_of_package_root(),'data/plots')

@make_output_path_obj
def get_abs_model_training_data_folder_path():
    return os.path.join(get_abs_path_of_package_root(),'data/csv_files/model_training_history')

@make_output_path_obj
def get_patch_model_training_data_folder_path():
    return os.path.join(get_abs_model_training_data_folder_path(), "patch_model")

@make_output_path_obj
def get_surface_points_model_training_data_folder_path():
    return os.path.join(get_abs_model_training_data_folder_path(), "surface_points_model")

@make_output_path_obj
def get_patch_model_training_data_file_abs_path_by_model_name(name):
    return os.path.join(get_patch_model_training_data_folder_path(), name + ".csv")
