import os
import keras
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.path_funcs import *
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
from utils.utils import get_top_N_largest_nums_indices_in_list
from utils.path_funcs import *
from data_processing.data_organized import collect_csv_files_into_one_df, get_training_and_testing_data_for_patch_model


class PatchClassificationModel:
    def __init__(self, NNShape=[], diamond=False, regularizer=False, name=""):
        
        if name:
            try:
                self.name = name
                model_path = get_abs_saved_patch_models_folder_path_with_model_name(name)
                training_history_csv_path = get_patch_model_training_data_file_abs_path_by_model_name(name)
                self.model = keras.models.load_model(model_path)
                colnames = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
                self.history = pd.read_csv(training_history_csv_path,names=colnames, header=0)
                
            except FileNotFoundError as error:
                print(f"{error}")
        
        else:
            
            if NNShape:
                total_layers = [layers.Input(shape=(3,))]
                for num_of_neurons in NNShape:
                    total_layers.append(layers.Dense(num_of_neurons, activation='relu'))
                
                if regularizer:
                    total_layers.append(layers.Dense(96, activation='softmax', kernel_regularizer='l2'))
                else:
                    total_layers.append(layers.Dense(96, activation='softmax'))
                
                self.model = keras.Sequential(total_layers)
                self.history = []
                if diamond:
                    shape_str = ""
                    for i in NNShape:
                        shape_str = shape_str + str(i) + "-"
                    
                    self.settings = {"shape":f"{shape_str}"}

                else:
                    self.settings = {"shape":f"{num_of_neurons}-"*len(NNShape)}


    def compile(self,opt, loss_, metrics_):
        self.settings['optimizer'] = opt._name
        self.settings['loss'] = loss_
        self.settings['metrics'] = metrics_

        self.model.compile(optimizer=opt, loss=loss_, metrics=metrics_)

    def train(self, data, epochs_=5, batch_size_=256, verbose_=1):
        X_train, X_test, Y_train, Y_test = data
        self.history = self.model.fit(X_train, Y_train, epochs=epochs_, shuffle=True,batch_size=batch_size_, validation_data=(X_test, Y_test), verbose=verbose_)
        self.settings["epochs"] = epochs_
        self.settings["shuffle"] = True
        self.settings["batch_size"] = batch_size_

    def predict(self, point, expected_patch):
        tensor_point = tf.convert_to_tensor([point])
        predicted_patch = np.argmax(self.model.predict(tensor_point))
        print(f"expected patch: {expected_patch}")
        print(f"predicted patch: {predicted_patch}")
        return [expected_patch, predicted_patch]
    
    def plot(self,func='loss',name='', save=False,add_to_title='' ,ext='jpg', show=True):

        title = f"{func} over {self.settings['epochs']} epochs"
        data1 = self.history.history[func]
        data2 = self.history.history[f'val_{func}']
        lowest_point1 = (len(data1)-1, data1[-1])
        lowest_point2 = (len(data2)-1, data2[-1])


        if func != 'loss':
            title = f"Accuracy over {self.settings['epochs']} epochs"
        
        if add_to_title != '':
            title = title + "\n" + add_to_title

        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticks(np.arange(1e-3, 2e+0, step=0.05))
        ax.set_yticklabels([str(i) for i in np.arange(1e-3, 2e+0, step=10)])
        ax.set_yscale("log")
        # ax.set_ylim(bottom=1e-3,top=2e+0)

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
        ax.set_xticks(np.arange(0,self.settings['epochs']+1, 5))
        ax.set_ylabel(func)
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
        plt.tick_params(axis='y', which='minor')
        
        if save:
                name = name + "-" + self.__create_file_name()
                plot_path = os.path.join(get_abs_path(get_relative_saved_plots_folder_path()),f"patch_model/{name}.{ext}")
                plt.savefig(plot_path)

        if show:
            plt.show()



    def save_(self, name="patch_model"):
        name = name + "-" + self.__create_file_name()
        path = os.path.join(get_abs_path(get_relative_saved_models_folder_path()),f"{name}.keras")
        self.model.save(path)
    
    def save_training_and_validation_data(self, name="patch_model"):
        name = name + "-" + self.__create_file_name()
        path = os.path.join(get_abs_path(get_relative_data_folder_path()),f"model_training_history/patch_model/{name}.csv")
        df = pd.DataFrame(self.history.history)
        with open(path, mode='w') as f:
            df.to_csv(f)

    def print_settings(self):
        for key, value in self.settings.items():
            print(f"{key}:{value}")
    
    def __create_file_name(self):
        return f"shape-{self.settings['shape']}bs-{self.settings['batch_size']}"



    
class Experiment:

    def __init__(self, nl,NN, list_epochs, list_batch_sizes,list_optimizers, diamond=False, regularizer=False):
        self.nums_layers = nl
        self.list_num_neurons_per_layer = NN
        self.list_epochs = list_epochs
        self.list_batch_sizes = list_batch_sizes
        self.list_optimizers = list_optimizers
        self.regularizer = regularizer
        self.combinations = self.create_combinations_of_settings()

        if diamond:
            pass
        if regularizer:
            print("weights (kernel) regularizer L2 will be used for output layer")

    def create_combinations_of_settings(self):
        return list(itertools.product(self.nums_layers,
                                      self.list_num_neurons_per_layer,
                                      self.list_epochs,
                                      self.list_batch_sizes,
                                      self.list_optimizers))

    def run(self,data, save=False, name="patch-model-experiment"):
        X_train, X_test, Y_train, Y_test = data
        combinations_of_settings = self.create_combinations_of_settings()

        for setting_ in combinations_of_settings:

            # name_ = self.create_file_name_from_settings(setting_)
            print("current settings: ", setting_)
            number_of_layers, number_of_neurons, epochs_, batch_size_, optimizer_ = setting_
            NNShape = [number_of_neurons] * number_of_layers
            print("NNSHAPE = ", NNShape)
            patch_model = PatchClassificationModel(NNShape=NNShape, regularizer=self.regularizer)
            
            patch_model.compile(opt=optimizer_, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'])
            patch_model.train((X_train, X_test, Y_train, Y_test),epochs_, batch_size_, verbose_=1)

            
            if save:
                patch_model.plot(add_to_title="loss function: sparse_categorical_crossentropy" ,save=save,show=False, name=name)
                patch_model.save_(name=name)
                patch_model.save_training_and_validation_data(name=name)

    def __str__(self):
        return f"The Experiment will try all different combinations of the following:\n\
        Number of Layers:{self.nums_layers}\n\
        Number of neurons per layer:{self.list_num_neurons_per_layer}\n\
        Number of epcohs:{self.list_epochs}\n\
        Number of batches:{self.list_batch_sizes}\n\
        List of optimizers:{self.list_optimizers}\n"
        
# class PatchModelWeightsTrained

class Utils:

    @classmethod
    def check_which_subsequent_guesses_are_correct(cls, path):



        X_train, X_test, Y_train, Y_test = get_training_and_testing_data_for_patch_model(amount=1, split=0.2, random_state=1)
        patch_model = tf.keras.models.load_model(path)

        results = { 
                    str(i):0 for i in range(97)
                    }
        # start_ = start
        # end_ = end
        wrong_predictions = 0
        number_of_samples = len(X_test)
        print("number of samples: ", number_of_samples)

        predictions = patch_model.predict(X_test)
        print("len of predictions: ", len(predictions))
        predictions_and_index_list = [get_top_N_largest_nums_indices_in_list(i,N=96) for i in predictions]
        
        expectations = Y_test
        for i in range(len(predictions_and_index_list)):
            expected_patch = Y_test[i]
            position = 0
            for prediction in predictions_and_index_list[i].keys():
                if int(prediction) == int(expected_patch):
                    if position != 0:
                        wrong_predictions += 1
                    results[str(position)] += 1
                    continue
                position += 1
        # print("percentage of mistakes: %2.2f %% ()" %(wrong_predictions*100/number_of_samples))
        print(f"percentage of mistakes: {(wrong_predictions/number_of_samples):2.2%} ({wrong_predictions})")
        for key, value in results.items():
            print(f"percentage of guess number {key} being correct: {(value/wrong_predictions):2.2%} {(value)}")



    def train_a_model_with_output_layer_weights_affected_by_number_of_samples_per_patch(shape):
        patch_model = PatchClassificationModel(NNShape=shape)
        last_layer = patch_model.model.layers[-1]
        print("model.layers: ", patch_model.model.layers)


    # def l1_reg(weight_matrix):
    #     return 0.01 * ops.sum(ops.absolute(weight_matrix))