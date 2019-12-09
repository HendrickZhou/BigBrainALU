import tensorflow as tf
import numpy as np
import pickle
import sys, os
from tools.utils import *

DATASET_EXT = "batch"
default_path = default_dataset_path()

def input_fn(data_dir,
             drop_remainder = None,
             *args):
    # relative path of dataset batch files
    batch_file_names = get_files_with_extension(data_dir, DATASET_EXT)
    dataset = tf.data.Dataset.from_tensor_slices(batch_file_names)
    # apply the dataset parsing function
    dataset = dataset.map(lambda x : tf.py_function(func=parse_file_fn, inp=[x], Tout=tf.string), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(2)
    dataset = dataset.prefetch(2)
    return dataset



def parse_filename_to_data(filename):
    """
    convert filename to dict of data and label
    """
    with open(filename, "rb") as f: 
        batch = pickle.load(f)
    return list(batch.values())

def preprocess(data, label):
    """
    preprocess datas
    """
    return data, label

def parse_file_fn(ds_elem) -> str:
    """
    The union preprocessing functions,
    directly used in dataset.map() function
    """
    ds_elem_str = ds_elem.numpy().decode('utf-8')
    return preprocess(*parse_filename_to_data(ds_elem_str))

    
if __name__ == "__main__":
    print("starting running")
    dataset = input_fn(default_path)
    for d in dataset:
        print(d)
    print("done running")
