from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from tools.utils import *
from tools.dataset import *

BITS = 6
OPS_BITS = 4

checkpoint_dir = lambda idx: str(default_train_sum_path()) + "/checkpoint/model_{}/".format(idx)
checkpoint_path = "cp-{epoch:04d}.ckpt"
tensorboard_path = lambda idx: str(default_train_sum_path()) + "/summary/model_{}/".format(idx)

def toy_net(layers, batch_size = 1):
    input_dims = 2*BITS + OPS_BITS
    output_dims = 1
    model = keras.Sequential()
    model.add(keras.layers.Dense(55, input_shape=(input_dims,)))
    for unit in layers:
        model.add(keras.layers.Dense(units=unit, activation='relu'))
    model.add(keras.layers.Dense(units=output_dims, activation='relu'))
    return model

# @tf.function
def train(model, dataset, idx):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_path,
                                                     monitor = "val_acc",
                                                     verbose=1,
                                                     save_best_only = True,
                                                     save_weights_only=True,
                                                     mode = "auto",
                                                     save_freq=100)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
                                                 update_freq = 100)
    model.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = model.fit(
        dataset, 
        steps_per_epoch = 100, #600
        epochs=2, # 20
        callbacks = [cp_callback, tb_callback]
        # validation_data=val_dataset.repeat(), 
        # validation_steps=2
    )
    # for x,y in dataset:
    #     with tf.GradientTape() as tape:
    #         prediction=model(x)
    #         loss = loss_fn(prediction, y)
    #     gradients = tape.gradient(loss, model.trainable_variable)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))





layers = [55, 50, 40, 35, 30, 25, 10, 15]

model = toy_net(layers)
dataset = input_fn(str(default_path / "alu_6.csv"), 16, [True for i in range(1)] + [False for i in range(4)])

idx = 1
import os
if not os.path.exists(checkpoint_dir(idx)):
    os.makedirs(checkpoint_dir(idx))
if not os.path.exists(tensorboard_path(idx)):
    os.makedirs(tensorboard_path(idx))
train(model, dataset, idx)