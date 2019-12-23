import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from .report_models import restore,evaluate

checkpoint_dir = lambda idx: str(default_train_sum_path()) + "/checkpoint_part1/model_{}/".format(idx)
tensorboard_path = lambda idx: str(default_train_sum_path()) + "/summary_part1/model_{}/".format(idx)

checkpoint_train_path = "train-001.hdf5"
checkpoint_valid_path = "valid-001.hdf5"

model_saved_path = str(default_train_sum_path()) + "/models_test/"
model_saved_name = lambda idx: model_saved_path + "model_{}.h5".format(idx)


def cal_acc(path):

