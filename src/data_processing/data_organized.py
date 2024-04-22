import os
import pandas as pd
import tensorflow as tf
from utils.path_funcs import *

def collect_all_dfs_into_one():
    data_csv_files_path = get_abs_path(get_relative_data_folder_path())
    all_csv_files = get_list_of_elements_in_dir(data_csv_files_path)
    dfs = []
    col_names = ["x", "y", "z", "patch", "sp1", "sp2", "sd"]
    # READY_CSV = pd.read_csv(os.path.join(data_csv_files_path, all_csv_files[0]),  names=col_names)
    # print(READY_CSV['patch'].unique())
    uniques = []
    for i in all_csv_files:
        # path to file Points_0_0_0.csv
        path = os.path.join(data_csv_files_path, i)#all_csv_files[-1])
        READY_CSV = pd.read_csv(path, names=col_names)
        dfs.append(READY_CSV)
        # for i in READY_CSV['patch'].unique():
        #     uniques.append(i)
        # print(READY_CSV['patch'].unique())

    # print(min(uniques), max(uniques))
    # uniques = list(set(uniques))
    # uniques.sort()
    # print(uniques)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values(by="patch")
    # correction of some data points
    final_df = final_df[(final_df[:,4]>=0)&(final_df[:,4]<=1)&(final_df[:,5]>=0)&(final_df[:,5]<=1)]
    # print(final_df.head())
    # print(final_df.tail())
    return final_df

def split_df_based_on_patch(df):
    dfs = []
    for _, d in df.groupby('patch'):
        dfs.append(d)
    
    return dfs

def split_dfs_for_training_testing_and_recombine(dfs, amount=1, split=0.2):
    train = []
    test = []
    for df in dfs:

        length = df.shape[0]
        if amount > 0 and amount != 1:
            limit_amount = int(length*amount)
            df = df.iloc[:limit_amount]
        limit_split = int(length*split)

        training_df = df.iloc[limit_split:,:]
        train.append(training_df)

        testing_df = df.iloc[:limit_split,:]
        test.append(testing_df)

    train_final = pd.concat(train, ignore_index=True)
    test_final = pd.concat(test, ignore_index=True)

    return train_final, test_final

def get_xys_sp1_sp2_sd_from_df(df):
    x = df["x"].tolist() 
    y = df["y"].tolist() 
    z = df["z"].tolist()
    patch = df["patch"].tolist()
    sp1 = df["sp1"].tolist() 
    sp2 = df["sp2"].tolist() 
    sd = df["sd"].tolist()

    return x, y, z, patch, sp1, sp2, sd

# print(get_xys_sp1_sp2_sd_from_df(READY_CSV))

def create_training_data(df, amount=1,split=0.2):
    
    dfs = split_df_based_on_patch(df)
    training_df, testing_df = split_dfs_for_training_testing_and_recombine(dfs,amount=amount, split=split)


    training_df = training_df.sample(frac=1)
    x_trn, y_trn, z_trn, patch_trn, sp1_trn, sp2_trn, sd_trn = get_xys_sp1_sp2_sd_from_df(training_df)
    X_train = [(x_trn[i], y_trn[i], z_trn[i]) for i in range(training_df.shape[0])]
    Y_train = [patch_trn[i] for i in range(training_df.shape[0])]
    x_tst, y_tst, z_tst, patch_tst, sp1_tst, sp2_tst, sd_tst = get_xys_sp1_sp2_sd_from_df(testing_df)
    X_test = [(x_tst[i], y_tst[i], z_tst[i]) for i in range(testing_df.shape[0])]
    Y_test = [patch_tst[i] for i in range(testing_df.shape[0])]
    
    return X_train, X_test, Y_train, Y_test

def get_training_and_testing_data(amount=1):
    # print("in get_training_and_testing_data")
    df = collect_all_dfs_into_one()
    # print(df.loc[(df['x'] == 1.8184613737638664) & (df['y'] == 2.893173926071881) & (df['z'] == 1.96274536399414)])
    # dfs = split_dfs_based_on_patch(df)
    # split_dfs_for_training_testing_and_recombine()
    X_train, X_test, Y_train, Y_test = create_training_data(df, amount=amount)
    X_train, X_test, Y_train, Y_test = tf.convert_to_tensor(X_train),tf.convert_to_tensor(X_test),tf.convert_to_tensor(Y_train),tf.convert_to_tensor(Y_test)
    return X_train, X_test, Y_train, Y_test

