import numpy as np
import pickle
import sys, os
from tools.utils import *

DATASET_EXT = "batch"
default_path = default_dataset_path()

def extract_batches(path = default_path):
    batch_names = get_files_with_extension(path, DATASET_EXT)
    for name in batch_names:
        data, label = extract_single_batch(name)
        print(data[:10])
        print(label[:10])

def extract_single_batch(name):
    with open(name, "rb") as f:
        batch = pickle.load(f)
    data = batch['data']
    label = batch['label']
    return data, label


if __name__ == "__main__":
    extract_batches()
