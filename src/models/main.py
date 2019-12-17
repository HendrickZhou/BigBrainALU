import tensorflow as tf
from capacity.cap_estimate import *
from .toy_model import *

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
    print(cap_estimate(l, 20, 1))


for i, l in enumerate(layers):
    train_on(l, 16, 1, new_ds(), i+1)
