import os
import keras
import time
import random
import argparse
import numpy as np
import pandas as pd
import keras.losses
import keras.optimizers
import tensorflow as tf
import moviepy.editor as mp
import matplotlib.pyplot as plt
import src.utils.utils as utils
from matplotlib.animation import PillowWriter
from keras.utils import to_categorical
from src.model_training.patch_model_settings import Experiment
from src.model_training.patch_model_settings import PatchClassificationModel
from src.model_training.patch_model_settings import PatchModelUtils
from src.model_training.surface_points_model_settings import SurfacePointsModel, SurfacePointsModelForOnePatch
from src.data_processing.data_organized import get_training_and_testing_data_for_patch_model, collect_csv_files_into_one_df,\
      split_df_based_on_patch,create_patch_model_training_data, plot_data, plot_bar_points_per_patch,\
          get_N_random_points_per_patch_for_patch_model_training,get_training_and_testing_data_and_sample_weights_for_patch_model,\
          get_training_and_testing_data_for_surface_point_model_for_one_patch
from src.utils.path_funcs import get_abs_saved_models_folder_path, get_abs_path, get_abs_raw_data_folder_path, get_abs_path_of_package_root
# from model_training.patch_model_settings import Utils
from tests.unit_tests.add_L2_regularizer_to_output_layer_of_patch_classification_model_test import *
from src.utils.decorators import cprofile_function
# from ...src.model_training.patch_model_settings import PatchModelUtils

import cProfile
import pstats
import io
from contextlib import redirect_stdout


# 10k points per patch = 960000 | full dataset = 8 960 000
# when amount == 0 takes full dataset
# X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
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

# layers1 = [2]
# neurons1 = [512]#, 1024, 2048]
# opt = tf.keras.optimizers.Adam()
# # exp = Experiment(nl=layers1,NN=neurons1, list_epochs=[1], list_batch_sizes=[64],list_optimizers=[opt], regularizer=False)
# # print(exp.create_combinations_of_settings())
# # exp.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_1_epochs-testttt")

# exp2 = Experiment(nl=layers1,NN=neurons1, list_epochs=[2000], list_batch_sizes=[64],list_optimizers=[opt], regularizer=True)
# # print(exp.create_combinations_of_settings())
# exp2.run((X_train, X_test, Y_train, Y_test),save=True, name="patch_model_2000_epochs_regularizer")

# neurons1 = [512, 1024, 2048, 4096]
# for i in neurons1:

#     name1 = f"patch_model_rand_sample_0.1-shape-{i}-{i}-bs-64"
#     name2 = f"patch_model_rand_sample_0.1_weights_regularizer-shape-{i}-{i}-bs-64"

#     model1 = PatchClassificationModel(name=name1)
#     model1.plot(name=name1, show=True, loaded_model=True, save=True)
#     model2 = PatchClassificationModel(name=name2)
#     model2.plot(name=name2, show=True, loaded_model=True, save=True)

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


def train_new_patch_model_raw(shape=None, name=None, epochs=100, verbose=None, regularizer=False, data_set_amount=0.1): #default: shape=None, name=None, epochs=100
    if verbose is not None:
        print(f"Now training a raw new patch model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
    


    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=data_set_amount, split=0.2, random_state=1) # gets training data for the whole dataset
    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(NNShape=shape,regularizer=regularizer)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=False)
    patch_model.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name)

def train_new_surface_points_model_raw(shape=None, name=None, epochs=100, verbose=None, regularizer=False, data_set_amount=0.1): #default: shape=None, name=None, epochs=100
    if verbose is not None:
        print(f"Now training a raw new surface points model.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
    
    X_train_points, Y_train_sp, X_test_points, Y_test_sp = get_training_and_testing_data_for_surface_point_model_for_one_patch(patch=25,amount=data_set_amount, split=0.2, random_state=1)
    print(
        f"""
        X_train_points.shape = {X_train_points.shape}
        Y_train_sp.shape = {Y_train_sp.shape}
        X_test_points.shape = {X_test_points.shape}
        Y_test_sp.shape = {Y_test_sp.shape}
        """
    )
    opt = tf.keras.optimizers.Adam()
    sp_model = SurfacePointsModelForOnePatch(NNShape=shape,regularizer=regularizer)
    sp_model.compile(opt=opt, loss_="mae", metrics_=["mae"])
    sp_model.train((X_train_points, X_test_points,  Y_train_sp, Y_test_sp), epochs, batch_size_=64, verbose_=verbose, name=name)


def retrain_existing_patch_model_raw(name=None, epochs=100, verbose=None,data_set_amount=0.1):
    if verbose is not None:
        print(f"Now training an existing patch model.")
        print(f"Epochs: {epochs}, name: {name}")
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=data_set_amount, split=0.2, random_state=1) # gets training data for the whole dataset
    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(name=name)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=False)
    patch_model.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name)


def train_new_patch_model_with_sample_weights(shape=None, name=None, epochs=100,verbose=None, regularizer=False, data_set_amount=0.1):
    if verbose is not None:
        print(f"Now training a new patch model with sample_weights.")
        print(f"Model shape: {shape}, Epochs: {epochs}, Name: {name}")
    X_train, X_test, Y_train, Y_test, sample_weights_for_training_data = get_training_and_testing_data_and_sample_weights_for_patch_model(amount=data_set_amount, split=0.2, random_state=1)
    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(NNShape=shape, regularizer=regularizer)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=True)
    patch_model.train((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name, sample_weights=sample_weights_for_training_data)


def retrain_existing_patch_model_with_sample_weights(name=None, epochs=100, verbose=None, data_set_amount=0.1):
    if verbose is not None:
        print(f"Now training an existing patch model with sample_weights.")
        print(f"epochs: {epochs}, name: {name}")
    X_train, X_test, Y_train, Y_test, sample_weights_for_training_data = get_training_and_testing_data_and_sample_weights_for_patch_model(amount=data_set_amount, split=0.2, random_state=1)
    opt = tf.keras.optimizers.Adam()
    patch_model = PatchClassificationModel(name=name)
    patch_model.compile(opt=opt, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'], sample_weight=True)
    patch_model.retrain_existing_model((X_train, X_test, Y_train, Y_test), epochs, batch_size_=64, verbose_=verbose, name=name, sample_weights=sample_weights_for_training_data)


def train(args):
    shape = args.shape
    epochs = args.epochs
    name = args.name
    regularizer = args.regularizer
    verbose = args.verbose
    data_set_amount = args.data_set
    if args.model_type == "p":
        if args.weights:
            train_new_patch_model_with_sample_weights(shape=shape, name=name, epochs=epochs, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount)
        else:
            train_new_patch_model_raw(shape=shape, name=name, epochs=epochs, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount)
        
    elif args.model_type == "sp":
        if args.weights:
            pass
        else:
            train_new_surface_points_model_raw(shape=shape, name=name, epochs=epochs, verbose=verbose, regularizer=regularizer,data_set_amount=data_set_amount)

def retrain(args):
    if args.model_type == "p":
        epochs = args.epochs
        name = args.name
        verbose = args.verbose
        data_set_amount = args.data_set

        print(args)
        if args.weights:
            retrain_existing_patch_model_with_sample_weights(name=name, epochs=epochs, verbose=verbose,data_set_amount=data_set_amount)
        else:
            retrain_existing_patch_model_raw(name=name, epochs=epochs, verbose=verbose,data_set_amount=data_set_amount)
        
    elif args.model_type == "sp":
        pass

def plot_model_perfomance_for_all_patches(patch_model_name, plot_name):
    X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset
    utils.bar_plot_patch_model_performance_for_all_patches(patch_model_name=patch_model_name, data=(X_train, X_test, Y_train, Y_test), plot_name=plot_name)


def plot_ppp(args):
    print(args)
    plot_model_perfomance_for_all_patches(patch_model_name=args.model_name, plot_name=args.plot_name)

def main():
    general_parser = argparse.ArgumentParser(description='thesis_sourcecode program.')
    subparsers = general_parser.add_subparsers(required=True)

    # create the parser for the "train" command
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-s','--shape', type=int,nargs='+', help='<Required> Set Number of hidden layers and their nodes', required=True)
    train_parser.add_argument('-mt','--model-type', type=str, help='<Required> Set type of model to train', required=True)
    train_parser.add_argument('-w','--weights', help='<Optional> Set whether to train model with sample_weights', action='store_true')
    train_parser.add_argument('-ds','--data-set',type=float, help='<Optional> Set how much of the dataset to use for training. 0 means use full dataset. default is 0.1')
    train_parser.add_argument('-r','--regularizer', help='<Optional> Set whether to train model with an L2 regularizer on the output layer', action='store_true')
    train_parser.add_argument('-e','--epochs', type=int,help='<Required> Set number of epochs to train the new model', required=True)
    train_parser.add_argument('-n','--name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    train_parser.add_argument('-v', '--verbose',action='store_true')
    train_parser.set_defaults(func=train)

    # create the parser for the "retrain" command
    retrain_parser = subparsers.add_parser('retrain')
    # train_parser.add_argument('-s','--shape', type=int,nargs='+', help='<Required> Set Number of hidden layers and their nodes', required=True)
    retrain_parser.add_argument('-mt','--model-type', type=str, help='<Required> Set type of model to train', required=True)
    retrain_parser.add_argument('-w','--weights', help='<Optional> Set whether to train model with sample_weights', action='store_true')
    retrain_parser.add_argument('-ds','--data-set',type=float, help='<Optional> Set how much of the dataset to use for training. 0 means use full dataset. default is 0.1')
    retrain_parser.add_argument('-e','--epochs', type=int,help='<Required> Set number of epochs to train the new model', required=True)
    retrain_parser.add_argument('-n','--name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    retrain_parser.add_argument('-v', '--verbose',action='store_true')
    retrain_parser.set_defaults(func=retrain)


    plot_ppp_parser = subparsers.add_parser('plot_ppp')
    plot_ppp_parser.add_argument('-mn','--model-name', type=str,help='<Required> Set name-prefix for the training history and model files', required=True)
    plot_ppp_parser.add_argument('-pn','--plot-name', type=str,help='<Required> Set name-prefix for the plot name and save it')
    plot_ppp_parser.set_defaults(func=plot_ppp)


    parse_args_output = general_parser.parse_args()
    parse_args_output.func(parse_args_output)



@cprofile_function("testingProfiler2_as_decorator")
def profiling():
    print("Hello world!")
    print(2**9)

if __name__ == "__main__":
    # utils.print_avg_last_20_training_epochs_with_std()
    main()

    # X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=0.1, split=0.2, random_state=1) # gets training data for the whole dataset

    
    # profiling()

    # # opt = tf.keras.optimizers.Adam()
    # # ###############
    # model_names = ["patch_model_rand_sample_0.1--shape-512-512-bs-64-200-epochs",
    #                "patch_model_rand_sample_0.1_weights_regularizer-shape-512-512-bs-64",
    #                "patch_model_rand_sample_0.1_sample_weights-shape-512-512-bs-64"]
    # # model_names = ["patch_model_rand_sample_0.1_sample_weights-shape-512-512-bs-64"]
    # utils.bar_plot_patch_model_performance_for_all_patches_for_multiple_models(patch_models_names_list=model_names,
    #             data=(X_train, X_test, Y_train, Y_test))
    ###############
    # model_name = "surface_points_model_test_patch_25"
    # # model_name = "patch_model_2000_epochs-"
    # # utils.plot_training_history(model_name=model_name,model_type="sp", plot_smooth=False)
    # # loaded_patch_model = PatchClassificationModel(name=model_name)
    # loaded_patch_model = SurfacePointsModelForOnePatch(name=model_name)
    # X_train_points, Y_train_sp, X_test_points, Y_test_sp = get_training_and_testing_data_for_surface_point_model_for_one_patch(patch=25,amount=0, split=0.2, random_state=1)
    # print(loaded_patch_model.predict(X_test_points))
    # print(Y_test_sp)

    # utils.plot_training_history(model_name=model_name, model_type="sp")

    pass