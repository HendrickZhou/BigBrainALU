from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from tools.utils import *
from tools.dataset import *

BITS = 6
OPS_BITS = 4

def toy_net(batch_size = 1):
    input_dims = 2*BITS + OPS_BITS
    output_dims = 1
    model = keras.Sequential([
        keras.layers.Dense(50, input_shape=(input_dims,)),
        keras.layers.Dense(units=40, activation='relu'),
        keras.layers.Dense(units=35, activation='relu'),
        keras.layers.Dense(units=30, activation='relu'),
        keras.layers.Dense(units=25, activation='relu'),
        keras.layers.Dense(units=20, activation='relu'),
        keras.layers.Dense(units=15, activation='relu'),
        keras.layers.Dense(units=output_dims, activation='sigmoid')
    ])

    return model

    # define the model
    # input = layers.Input(shape=(input_dims,1,1000))
    # x = input
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # x = layers.Dense(20, activation=activa)(x)
    # output = layers.Dense(1, activation=activa)(x)
    
    # return models.Model(input, output, name="toy_net")


# @tf.function
def train(model, dataset):

    model.compile(optimizer='adam', 
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        dataset, 
        steps_per_epoch=1000,
        epochs=1,
        # validation_data=val_dataset.repeat(), 
        # validation_steps=2
    )
    # for x,y in dataset:
    #     with tf.GradientTape() as tape:
    #         prediction=model(x)
    #         loss = loss_fn(prediction, y)
    #     gradients = tape.gradient(loss, model.trainable_variable)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

model = toy_net()
dataset = input_fn(default_dataset_path())
# for data in dataset:
#     print(data)
train(model, dataset)

# start teh training




# 
