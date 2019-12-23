import tensorflow as tf
from capacity.cap_estimate import *
from .toy_model import *
from tools.dataset import *

# layers

layers_1 = [12, 11, 10, 9, 8]
layers_2 = [12, 8, 8, 8]
layers_3 = [12, 8, 8]
layers_4 = [11, 9, 9, 3]
layers_5 = [11, 9, 8]
layers_6 = [10, 10, 10]
layers_7 = [10, 10]
layers_8 = [9,9,8]
layers_9 = [9, 4]
layers_10 = [8,8]
layers_11 = [7, 7, 7]
layers_12 = [7, 6]
layers_13 = [6,6,6]
layers_14 = [6]
layers_15 = [5,5,5]
layers_16 = [5]
layers_17 = [4, 4, 4]
layers_18 = [4]
layers_19 = [3, 3, 3]
layers_20 = [2, 2]
layers = [layers_1, layers_2, layers_3, layers_4, layers_5, layers_6, layers_7, layers_8, layers_9, layers_10,layers_11, layers_12, layers_13, layers_14, layers_15, layers_16, layers_17, layers_18, layers_19, layers_20]

# capacity
for i, l in enumerate(layers):
    print("layer" + str(i))
    print(cap_estimate(l, 16, 1))

cp_callback = lambda idx : tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_path,
                                                 monitor = "acc",
                                                 verbose=1,
                                                 #save_best_only = True,
                                                 save_weights_only=True,
                                                 mode = "auto",
                                                 save_freq=100)
tb_callback = lambda idx : tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
                                                 update_freq = 100)


full_dataset = input_fn(str(default_path / "alu_6.csv"), 16, [True for i in range(1)] + [False for i in range(5)]).repeat()
for i, l in enumerate(layers):
    train_on(l, 16, 1, full_dataset, None, [cp_callback_no_valid, tb_callback_no_valid], i+1)
