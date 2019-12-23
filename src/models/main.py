import tensorflow as tf
from capacity.cap_estimate import *
from .toy_model import *
from tools.dataset import *

# layers

layers_1 = [50, 15, 11, 10, 9, 8]
layers_2 = [50, 15, 8, 8, 8]
layers_3 = [20, 10, 8, 8]
layers_4 = [20, 10, 9]
layers_5 = [20, 9]
layers_6 = [20]
layers_7 = [15, 10, 8]
layers_8 = [15, 10]
layers_9 = [15]
layers_10 = [13, 4]
layers_11 = [10, 7, 7]
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
    print(cap_estimate(l, 15, 1))

cp_callback = lambda idx : tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_path,
                                                 monitor = "acc",
                                                 verbose=1,
                                                 #save_best_only = True,
                                                 save_weights_only=True,
                                                 mode = "auto",
                                                 save_freq="epoch")
tb_callback = lambda idx : tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
                                                 update_freq = 100)


full_dataset = input_fn(str(default_path / "alu_6.csv"), 15, [True for i in range(1)] + [False for i in range(5)]).repeat()
#for i, l in enumerate(layers):
#   train_on(l, 15, 1, full_dataset, None, [cp_callback_no_valid, tb_callback_no_valid], i+1)
