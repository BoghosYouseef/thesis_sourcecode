import os
import keras
import time
import random
import pandas as pd
import keras.losses
import keras.optimizers
import numpy as np
import tensorflow as tf
from utils.utils import *
import moviepy.editor as mp
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from keras.utils import to_categorical
from model_training.patch_model_settings import Experiment
from model_training.patch_model_settings import PatchClassificationModel
from model_training.surface_points_model_settings import SurfacePointsModel
from data_processing.data_organized import get_training_and_testing_data_for_patch_model, collect_csv_files_into_one_df, split_df_based_on_patch,create_patch_model_training_data, plot_data, plot_bar_points_per_patch, get_N_random_points_per_patch_for_patch_model_training
from utils.path_funcs import get_abs_saved_models_folder_path, get_abs_path, get_abs_raw_data_folder_path, get_abs_path_of_package_root
from model_training.patch_model_settings import Utils


# 10k points per patch = 960000 | full dataset = 8 960 000
# when amount == 0 takes full dataset
X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0, split=0.2, random_state=1) # gets training data for the whole dataset
# X_train, X_test, Y_train, Y_test = get_N_random_points_per_patch_for_patch_model_training(10000, random_state=1, split=0.2)
# print(get_abs_path_of_package_root())
# collect_csv_files_into_one_df()
# df = collect_csv_files_into_one_df()
# X_train, X_test, Y_train, Y_test = create_patch_model_training_data(df, amount=0.1, split=0.2, random_state=1)
# X_train_points,X_train_patches, X_test_points,X_test_patches, Y_train_p1, Y_train_p2, Y_test_p1,Y_test_p2 = get_training_and_testing_data(amount=0.5, split=0.2, model=1) # gets training data for the whole dataset
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

layers1 = [2, 3, 4]
neurons1 = [256, 512, 1024, 2048, 4096]
opt = tf.keras.optimizers.Adam()
exp = Experiment(nl=layers1,NN=neurons1, list_epochs=[1000], list_batch_sizes=[64],list_optimizers=[opt], regularizer=True)
print(exp.create_combinations_of_settings())
exp.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_full_dataset_weights_regularizer")

# ###############
# NN_SHAPES = []
# starting_neurons = [96, 128]
# number_of_layers = [3, 4, 5, 6]
# for i in range(len(starting_neurons)):

#     nnshape = create_diamond_shape_using_powers_of_two(starting_num_neurons=starting_neurons[i],n=number_of_layers[i])
#     NN_SHAPES.append(nnshape)
#     # print("nnshape = ", nnshape)

# # for nn_shape in NN_SHAPES[1:]:
# nn_shape = create_diamond_shape_using_powers_of_two(starting_num_neurons=8, n=7)
# name="patch_model_rand_sample_0.1-"
# # # print("current nnshape training: ", nn_shape)
# patch_model = PatchClassificationModel(NNShape=[512, 512], regularizer=True)
# patch_model.compile(opt=opt,loss_="sparse_categorical_crossentropy",metrics_=['accuracy'])
# patch_model.train((X_train, X_test, Y_train, Y_test),epochs_=50, batch_size_=1024)
# patch_model.plot(show=True, save=False)
# patch_model.save_(name=name)
# patch_model.save_training_and_validation_data(name=name)
# data = collect_csv_files_into_one_df()
# data_per_patch = split_df_based_on_patch(data)
# total = 0
# for df in data_per_patch:
#     total += len(df.index)
#     print("for patch ", df["patch"].unique(), "there are  ", "{:,}###".format(len(df.index)), " total = ", total)
# print_training_results()

# data = collect_csv_files_into_one_df()
# X_train, X_test, Y_train, Y_test = get_N_random_points_per_patch_for_patch_model_training(10000, random_state=1, split=0.2)
# print("x_train.shape: ", X_train.shape)
# print("Y_train.shape: ", Y_train.shape)
# print("X_test.shape: ", X_test.shape)
# print("Y_test.shape: ", Y_test.shape)
# layers1 = [2]
# neurons1 = [512]#, 1024, 2048, 4096]
# opt = tf.keras.optimizers.Adam()
# exp = Experiment(nl=layers1,NN=neurons1, list_epochs=[5000], list_batch_sizes=[64],list_optimizers=[opt])
# print(exp.create_combinations_of_settings())
# exp.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_rand_sample_20k_points_per_patch_5k_epochs")
# path = "C:/UniversityImportantFiles/Master/semester 4/thesis_sourcecode/src/model_training/saved_models/patch_model_rand_sample_10k_points_per_patch-shape-512-512-bs-64.keras"
# # # Utils.check_which_subsequent_guesses_are_correct(path=path)
# model = PatchClassificationModel.load_model(path=path)
# trn_predictions = model.predict(X_train)
# tst_predictions = model.predict(X_test)
# wrongly_predicted_training_points = [X_train[i] for i in range(len(X_train)) if np.argmax(trn_predictions[i]) != Y_train[i]]
# wrongly_predicted_testing_points = [X_train[i] for i in range(len(X_test)) if np.argmax(tst_predictions[i]) != Y_test[i]]
# plot_data(training_data=wrongly_predicted_training_points, testing_data=wrongly_predicted_testing_points)
# list_of_wrong_guesses_trn = [i for i in X_train]

# def main():
    
#     loaded_patch_model = PatchClassificationModel(name="patch_model_rand_sample_10k_points_per_patch-shape-512-512-bs-64")
#     plot_training_history(loaded_patch_model, show=True)
#     pass

if __name__ == "__main__":
    main()