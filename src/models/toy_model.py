from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from tools.utils import *
from tools.dataset import *
from capacity.cap_estimate import *
BITS = 6
OPS_BITS = 3

checkpoint_dir = lambda idx: str(default_train_sum_path()) + "/checkpoint_part1/model_{}/".format(idx)
tensorboard_path = lambda idx: str(default_train_sum_path()) + "/summary_part1/model_{}/".format(idx)
checkpoint_path = "cp-{epoch:04d}.hdf5"
checkpoint_train_path = "train-{epoch:03d}.hdf5"
checkpoint_valid_path = "valid-{epoch:03d}.hdf5" #-{val_acc:.4f}

model_saved_path = str(default_train_sum_path()) + "/models_part1/"
model_saved_name = lambda idx: model_saved_path + "model_{}.h5".format(idx)


###############
# Model definition
###############
def toy_net(layers, batch_size = 1):
    input_dims = 2*BITS + OPS_BITS
    output_dims = 1
    model = keras.Sequential()
    model.add(keras.layers.Dense(layers[0], input_shape=(input_dims,)))
    for unit in layers[1:]:
        model.add(keras.layers.Dense(units=unit, activation='relu'))
    model.add(keras.layers.Dense(units=output_dims, activation='relu'))
    return model


#################
# Dataset importint
################
def new_ds(train = True):
    if train:
        dataset = input_fn(str(default_path / "3ops/alu_6.csv"), 16, [True for i in range(1)] + [False for i in range(5)])
        dataset = dataset.repeat()
    else:
        dataset = input_fn(str(default_path / "alu_6_valid.csv"), 16, [True for i in range(1)] + [False for i in range(5)])
    return dataset


##############
# Training definition
##############
# @tf.function
def train(model, train_set, valid_set, callbacks):
    """
    callbacks is list of callbacks
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    if valid_set:
        history = model.fit(
            train_set, 
            steps_per_epoch = 50, #600
            epochs=50, # 20
            callbacks = callbacks,
            validation_data=valid_set, 
            validation_steps=None,
            validation_freq=1,
        )
    else:
        history = model.fit(
            train_set, 
            steps_per_epoch = 50, #600
            epochs=300, # 20
            callbacks = callbacks,
        )
 
cp_callback_no_valid = lambda idx : tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_path,
                                                 monitor = "acc",
                                                 verbose=1,
                                                 #save_best_only = True,
                                                 save_weights_only=False,
                                                 mode = "auto",
                                                 save_freq="epoch")
tb_callback_no_valid =lambda idx : tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
                                                 update_freq = 10000)
cp_callback = lambda idx : tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_path,
                                                 monitor = "val_acc",
                                                 verbose=1,
                                                 #save_best_only = True,
                                                 save_weights_only=True,
                                                 mode = "auto",
                                                 save_freq="epoch")
tb_callback = lambda idx : tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
                                                 update_freq = 10000)


cp_callback_train_test = lambda idx: tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_train_path,
        moniter = "acc",
        verbose=1,
        save_best_only = True,
        save_weights_only = False,
        mode = "max",
        save_freq="epoch")
cp_callback_valid_test = lambda idx: tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_valid_path,
        moniter = "val_acc",
        verbose=1,
        save_best_only = True,
        save_weights_only = False,
        mode = "max",
        save_freq="epoch")
tb_callback_test = lambda idx : tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
        histogram_freq = 10,
        write_graph = True,
        write_images = True,
        update_freq = 10000)


#############
# Training Entry
############
import os
def train_on(layers, input_dim, output_dim, train_set, valid_set, callbacks_fn, idx):
    model = toy_net(layers)
    print("capacity of this dense net: {}".format(cap_estimate(layers,input_dim, output_dim)))
    if not os.path.exists(checkpoint_dir(idx)):
        os.makedirs(checkpoint_dir(idx))
    if not os.path.exists(tensorboard_path(idx)):
        os.makedirs(tensorboard_path(idx))
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    print("training on model:{}".format(idx))
    print(model.summary())
    callbacks = [cb_fn(idx) for cb_fn in callbacks_fn]
    train(model, train_set, valid_set, callbacks)
    model.save(model_saved_name(idx))

if __name__ == "__main__":
    layers = [55, 50, 40, 35, 30, 25, 20, 15]
    callbacks = [cp_callback_train_test,cp_callback_valid_test, tb_callback_test]
    train_on(layers, 16, 1, new_ds(True), new_ds(False), callbacks, 1)
    train_on(layers, 16, 1, new_ds(True), new_ds(False), callbacks, 2)
