import os
import keras
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from utils.path_funcs import get_relative_saved_models_folder_path, get_abs_path, get_relative_saved_plots_folder_path

class History:

    def __init__(self, history, settings):
        self.history = history
        self.settings = settings


class PatchClassificationModel:
    def __init__(self, num_layers, num_neurons_per_layer):
        total_layers = [layers.Input(shape=(3,))]
        for i in range(num_layers):
            total_layers.append(layers.Dense(num_neurons_per_layer, activation='relu'))
        
        total_layers.append(layers.Dense(96, activation='softmax'))
        
        self.model = keras.Sequential(total_layers)
        self.history = []
        self.settings = {}
        

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
        predicted_patch = list(self.model.predict(tensor_point)[0]).index(max(self.model.predict(tensor_point)[0]))
        print(f"expected patch: {expected_patch}")
        print(f"predicted patch: {predicted_patch}")
        return [expected_patch, predicted_patch]
    
    def plot(self,func='loss',name='',add_to_title='' ,ext='jpg', show=True):

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
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yscale("log")
        # ax.set_ylim(bottom=1e-4,top=1e+0)
        ax.set_xticks([int(len(data1)*0.5),int(len(data1)*0.75)])

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
        ax.grid(True) 
        ax.set_title(title)
        ax.set_xlabel("epochs")
        ax.set_xticks(np.arange(0,self.settings['epochs']+1, 5))
        ax.set_ylabel(func)

        
        if name != '':
            plot_path = os.path.join(get_abs_path(get_relative_saved_plots_folder_path()),f"{name}.{ext}")
            plt.savefig(plot_path)

        if show:
            plt.show()
    def save_(self, name="patch_model"):
        path = os.path.join(get_abs_path(get_relative_saved_models_folder_path()),f"{name}.keras")
        self.model.save(path)


    def print_settings(self):
        for key, value in self.settings.items():
            print(f"{key}:{value}")


    
class Experiment:

    def __init__(self, list_nums_layers,list_num_neurons_per_layer, list_epochs, list_batch_sizes,list_optimizers):
        self.list_nums_layers = list_nums_layers
        self.list_num_neurons_per_layer = list_num_neurons_per_layer
        self.list_epochs = list_epochs
        self.list_batch_sizes = list_batch_sizes
        self.list_optimizers = list_optimizers

        self.combinations = self.create_combinations_of_settings()

    def create_combinations_of_settings(self):
        return list(itertools.product(self.list_nums_layers,
                                      self.list_num_neurons_per_layer,
                                      self.list_epochs,
                                      self.list_batch_sizes,
                                      self.list_optimizers))
    
    def run(self,data):
        X_train, X_test, Y_train, Y_test = data
        combinations_of_settings = self.create_combinations_of_settings()

        for setting_ in combinations_of_settings:

            name_ = self.create_file_name_from_settings(setting_)
            number_of_layers, number_of_neurons, epochs_, batch_size_, optimizer_ = setting_
            patch_model = PatchClassificationModel(number_of_layers, number_of_neurons)
            patch_model.compile(opt=optimizer_, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'])
            patch_model.train((X_train, X_test, Y_train, Y_test),epochs_, batch_size_, verbose_=1)
            patch_model.plot(name=name_,add_to_title="loss function: sparse_categorical_crossentropy" ,show=False)
            patch_model.save_(name=name_)


    def create_file_name_from_settings(self, settings):
        number_of_layers, number_of_neurons, epochs_, batches_, optimizer_ = settings
        return f"nl-{number_of_layers}-nn-{number_of_neurons}-b-{batches_}-opt-{optimizer_._name}"

    def __str__(self):
        return f"The Experiment will try all different combinations of the following:\n\
        Number of Layers:{self.list_nums_layers}\n\
        Number of neurons per layer:{self.list_num_neurons_per_layer}\n\
        Number of epcohs:{self.list_epochs}\n\
        Number of batches:{self.list_batch_sizes}\n\
        List of optimizers:{self.list_optimizers}\n"
        



# model = keras.Sequential([
#     layers.Input(shape=(3,)),
#     layers.Dense(50, activation='relu'),
#     layers.Dense(50, activation='relu'),
#     layers.Dense(95, activation='softmax')
# ])

# model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])



