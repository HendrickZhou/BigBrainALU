import tensorflow as tf
import numpy as np
import pickle
import sys, os
from tools.utils import *

DATASET_EXT = "batch"
default_path = default_dataset_path()

def input_fn(csv_file_name,
             input_size,
             output_select, # expect a list
             drop_remainder = None,
             *args):
    # relative path of dataset batch files
    # batch_file_names = get_files_with_extension(data_dir, DATASET_EXT)
    # dataset = tf.data.Dataset.from_tensor_slices(batch_file_names)
    # dataset = dataset.shuffle(len(batch_file_names)).repeat()
    # apply the dataset parsing function
    # result = dataset.map(lambda x : tf.py_function(func=parse_file_fn, inp=[x], Tout=[tf.uint8, tf.uint8]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # reconstruct from the csv file
    # only support tf.int32 tf.int64...wtf
    args_data = tf.constant(input_size, dtype=tf.int32)
    select_cols = [i for i in range(input_size)]
    select_cols = select_cols + [i+input_size for i,v in enumerate(output_select) if v] # offset for select
    output_size = sum(output_select)
    # import pdb; pdb.set_trace()
    # a valid CSV DType (float32, float64, int32, int64, string)...wtf
    result = tf.data.experimental.CsvDataset(csv_file_name, 
                                             record_defaults = [tf.int32 for _ in range(input_size+output_size)],
                                             select_cols=select_cols)
    result = result.map(lambda *items: tf.stack(items))
    result = result.map(lambda item: split_data_label(item, args_data), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    result = result.shuffle(10000).batch(1000)
    result = result.cache().repeat().prefetch(100)
    return result

def split_data_label(element, input_len):
    # data = tf.slice(element, [0,], [data_len.numpy(),])
    data = element[:input_len.numpy()]
    # label = tf.slice(element, [data_len.numpy(),],[label_len.numpy(),])
    label = element[input_len.numpy():]
    return preprocess(data, label)

# def parse_filename_to_data(filename):
#     """
#     convert filename to dict of data and label
#     """
#     with open(filename, "rb") as f: 
#         batch = pickle.load(f)
#     data = batch["data"]
#     label = batch['label'][:,0]
#     return data,label

def preprocess(data, label):
    """
    preprocess datas
    """
    return data, label

# def parse_file_fn(ds_elem: tf.Tensor) -> str:
#     """
#     The union preprocessing functions,
#     directly used in dataset.map() function
#     """
#     ds_elem_str = ds_elem.numpy().decode('utf-8')
#     return preprocess(*parse_filename_to_data(ds_elem_str))
    
if __name__ == "__main__":
    print("starting running")
    # They fucking don't support path object, you have to convert it to string first
    dataset = input_fn(str(default_path / "alu_6.csv"), 16, [True] + [False for i in range(5)])
    def extract_dataset(dataset):
        for data, label in dataset:
            tf.print(data, output_stream=sys.stdout)
            tf.print(label, output_stream=sys.stdout)

    extract_dataset(dataset)
    print("done running")
