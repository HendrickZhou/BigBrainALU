import tensorflow as tf
from tools.utils import *
from tools.dataset import *
from models.toy_model import new_ds

checkpoint_dir = lambda idx: str(default_train_sum_path()) + "/checkpoint_test/model_{}/".format(idx)
tensorboard_path = lambda idx: str(default_train_sum_path()) + "/summary_test/model_{}/".format(idx)

checkpoint_train_path = "train-001.hdf5"
checkpoint_valid_path = "valid-001.hdf5"

model_saved_path = str(default_train_sum_path()) + "/models_test/"
model_saved_name = lambda idx: model_saved_path + "model_{}.h5".format(idx)

def restore(model_path, checkpoints_path):
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    #latest = tf.train.latest_checkpoint(checkpoints_path)
    model.load_weights(checkpoints_path)
    return model

def evaluate(model, ds):
    loss, acc = model.evaluate(valid, verbose = 1)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == "__main__":
    model_1 = restore(model_saved_name(1), checkpoint_dir(1)+checkpoint_valid_path)
    model_2 = restore(model_saved_name(2), checkpoint_dir(2)+checkpoint_train_path)
    valid = new_ds(False)
    loss, acc = model_1.evaluate(valid, verbose = 2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
