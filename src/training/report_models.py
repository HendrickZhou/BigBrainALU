import tensorflow as tf
from tools.utils import *
from tools.dataset import *
from models.toy_model import new_ds

checkpoint_dir = lambda idx: str(default_train_sum_path()) + "/checkpoint_part1/model_{}/".format(idx)
tensorboard_path = lambda idx: str(default_train_sum_path()) + "/summary_part1/model_{}/".format(idx)

checkpoint_train_path = "train-100.hdf5"
checkpoint_valid_path = "valid-100.hdf5"
checkpoint_no_valid = "cp-0100.ckpt"

model_saved_path = str(default_train_sum_path()) + "/models_test/"
model_saved_name = lambda idx: model_saved_path + "model_{}.h5".format(idx)

def restore(model_path, checkpoints_path, ckpt):
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    if ckpt:
        latest = tf.train.latest_checkpoint(checkpoints_path)
        model.load_weights(latest)
    else:
        model.load_weights(checkpoints_path)
    return model

def evaluate(model, ds):
    loss, acc = model.evaluate(valid, verbose = 1)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == "__main__":
    model_1 = restore(model_saved_name(1), checkpoint_dir(1),True)
    #model_2 = restore(model_saved_name(2), checkpoint_dir(2)+checkpoint_train_path)
    full_dataset = input_fn(str(default_path / "3ops/alu_6.csv"), 15, [True for i in range(1)] + [False for i in range(5)]).repeat()
    loss, acc = model_1.evaluate(full_dataset, verbose = 2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
