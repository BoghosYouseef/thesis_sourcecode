import keras
import time
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras import layers
from model_training.patch_model_settings import Experiment
from model_training.patch_model_settings import PatchClassificationModel
from data_processing.data_organized import get_training_and_testing_data, get_5patch_training_and_testing_data
from utils.path_funcs import get_relative_saved_models_folder_path, get_abs_path

# print(tf.test.gpu_device_name())
X_train, X_test, Y_train, Y_test = get_training_and_testing_data(amount=0.2) # gets training data for the whole dataset
# start_t = time.time()
########
# all_training_sets = get_5patch_training_and_testing_data() # gets training data for every 5 patches
# X_train1, X_test1, Y_train1, Y_test1 = all_training_sets[-1][0], all_training_sets[-1][1], all_training_sets[-1][2], all_training_sets[-1][3]
# X_train2, X_test2, Y_train2, Y_test2 = all_training_sets[1][0], all_training_sets[1][1], all_training_sets[1][2], all_training_sets[1][3]
########

# new_point1 = (-4.574666709326302,-1.3371642564998831,1.0224389644513838) # patch: 70
# tensor_point1 = tf.convert_to_tensor([new_point1])
# new_point2 = (2.0078873910406125,1.4705823248530572,1.100797831079947) # patch: 80
# tensor_point2 = tf.convert_to_tensor([new_point2])
# new_point3 = (1.9605308867214255,3.5670331055965865,1.8843864973655768) # patch: 0
# tensor_point3 = tf.convert_to_tensor([new_point3])

# list_nums_layers = [2, 4, 8, 12, 16]
# list_num_neurons_per_layer = [10, 25, 50, 80]
# list_epochs = [100]
# list_batch_sizes = [256]
# list_optimizers = [keras.optimizers.Adam(), keras.optimizers.SGD()]

# experiment = Experiment(list_nums_layers,list_num_neurons_per_layer, list_epochs, list_batch_sizes,list_optimizers)
# # print(experiment)
# data = (X_train, X_test, Y_train, Y_test)
# experiment.run(data)
# ####################################################
patch_model = PatchClassificationModel(2, 50)
optimizer = keras.optimizers.Adam()
patch_model.compile(opt=optimizer, loss_="sparse_categorical_crossentropy", metrics_=['accuracy'])
patch_model.train((X_train, X_test, Y_train, Y_test),epochs_=100, batch_size_=4096, verbose_=1)
patch_model.plot(name='test1', show=True)
# print("time in seconds: ", time.time()- start_t)
# patch_model.print_settings()
# # patch_model.save_(name="newest_patch_model")
# #########################################################
# for training_set in all_training_sets:
#     X_train, X_test, Y_train, Y_test = training_set[0],training_set[1], training_set[2],training_set[3]
#     patch_model.fit(X_train, Y_train, epochs=5, shuffle=True,batch_size=256, validation_data=(X_test, Y_test), verbose=1)
#     # test_loss = patch_model.evaluate(X_test, Y_test)
#     # print("="*10)
#     # print(f"for patches {min(Y_train)}-{max(Y_train)}")
#     # print(f"Test Loss: {test_loss[0]}")
#     # print(f"Test accuracy: {test_loss[1]}")
#     # print("*"*10)
#     position1 = list(patch_model.predict(tensor_point1)[0]).index(max(patch_model.predict(tensor_point1)[0]))
#     position2 = list(patch_model.predict(tensor_point2)[0]).index(max(patch_model.predict(tensor_point2)[0]))
#     position3 = list(patch_model.predict(tensor_point3)[0]).index(max(patch_model.predict(tensor_point3)[0]))
#     print("HERE SHOULD BE 70 =>>",position1)
#     print("HERE SHOULD BE 80 =>>",position2)
#     print("HERE SHOULD BE 0 =>>",position3)

# loaded_patch_model = keras.models.load_model(patch_model_path)
# new_point1 = (2.0552438953597987,3.6044697266812924,0.9179604756132997) # patch: 1
# new_point2 = (1.77110486944468,1.1710893561754108,-0.6230972347484396) # patch: 44
# new_point3 = (3.002373981743527,1.8823851567848222,-1.981317589643532) # patch: 16
