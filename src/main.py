import os
import keras
import time
import pandas as pd
import keras.losses
import keras.optimizers
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from model_training.patch_model_settings import Experiment
from model_training.patch_model_settings import PatchClassificationModel
from model_training.surface_points_model_settings import SurfacePointsModel
from data_processing.data_organized import get_training_and_testing_data_for_patch_model, collect_csv_files_into_one_df
from utils.path_funcs import get_relative_saved_models_folder_path, get_abs_path
from utils.utils import create_diamond_shape_using_powers_of_two, print_training_results
# X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
# X_train_points,X_train_patches, X_test_points,X_test_patches, Y_train_p1, Y_train_p2, Y_test_p1,Y_test_p2 = get_training_and_testing_data(amount=0.5, split=0.2, model=1) # gets training data for the whole dataset
# print(X_train[:5])
# print(Y_train[:5])
# print(X_test[:5])
# print(Y_test[:5])
# layers1 = [2, 4, 8, 12, 16]
# neurons1 = [10, 25, 50, 80]
# opt = tf.keras.optimizers.Adam()
# exp = Experiment(nl=layers1,NN=neurons1, list_epochs=[100], list_batch_sizes=[64],list_optimizers=[opt])
# print(exp.create_combinations_of_settings())
# exp.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_rand_sample_0.1")

# ###############
# NN_SHAPES = []
# starting_neurons = [4, 8, 16, 32, 64]
# number_of_layers = [2, 3, 4, 5, 6]
# for i in range(len(starting_neurons)):

#     nnshape = create_diamond_shape_using_powers_of_two(starting_num_neurons=starting_neurons[i],n=number_of_layers[i])
#     NN_SHAPES.append(nnshape)
#     # print("nnshape = ", nnshape)

# # for nn_shape in NN_SHAPES[1:]:
# nn_shape = create_diamond_shape_using_powers_of_two(starting_num_neurons=8, n=7)
# name="patch_model_rand_sample_0.1-"
# # # print("current nnshape training: ", nn_shape)
# patch_model = PatchClassificationModel(NNShape=[8192])
# patch_model.compile(opt=opt,loss_="sparse_categorical_crossentropy",metrics_=['accuracy'])
# patch_model.train((X_train, X_test, Y_train, Y_test),epochs_=100, batch_size_=64)
# patch_model.plot(show=False, save=True,name=name)
# patch_model.save_(name=name)
# patch_model.save_training_and_validation_data(name=name)
# collect_csv_files_into_one_df()
print_training_results()