import tensorflow as tf
from tools.utils import *
from tools.dataset import *
from models.toy_model import new_ds
from capacity.cap_estimate import *
import sys

checkpoint_dir = lambda idx: str(default_train_sum_path()) + "/checkpoint_part2/model_{}/".format(idx)
tensorboard_path = lambda idx: str(default_train_sum_path()) + "/summary_part2/model_{}/".format(idx)

checkpoint_train_path = lambda x:"train-{}.hdf5".format(x)
checkpoint_valid_path = lambda x:"valid-{}.hdf5".format(x)
checkpoint_no_valid = lambda x: "cp-0{}.hdf5".format(x)

model_saved_path = str(default_train_sum_path()) + "/models_part2/"
model_saved_name = lambda idx: model_saved_path + "model_{}.h5".format(idx)

def restore(model_path, checkpoints_path):
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    layers = []
    for layer in model.layers:
        layers.append(layer.get_output_at(0).get_shape().as_list()[1])
    print(cap_estimate(layers, 15, 1))
    model.load_weights(checkpoints_path)
    return model

def evaluate(model, ds):
    loss, acc = model.evaluate(valid, verbose = 1)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == "__main__":
    model_idx = sys.argv[1]
    cp_idx = sys.argv[2]
    model_1 = restore(model_saved_name(model_idx), checkpoint_dir(model_idx)+checkpoint_valid_path(cp_idx))
    full_dataset = input_fn(str(default_path / "3ops/alu_6.csv"), 15, [True for i in range(1)] + [False for i in range(5)])
    valid_set = new_ds(False)
    train_set = input_fn(str(default_path / "3ops/alu_6_train.csv"), 15, [True for i in range(1)] + [False for i in range(5)])
    loss, acc = model_1.evaluate(train_set, verbose = 0)
    print("Restored model, training accuracy: {:5.2f}%".format(100*acc))
    loss, acc = model_1.evaluate(valid_set, verbose = 0)
    print("Restored model, valid accuracy: {:5.2f}%".format(100*acc))
