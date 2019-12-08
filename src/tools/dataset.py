import numpy as np
import pickle
import sys, os

DATASET_EXT = "batch"

def get_files_with_extension(directory, extension):
    files = []
    for name in os.listdir(directory):
        if name.endswith(extension):
            files.append(f'{directory}/{name}')
    return files


def extract_batches(path):
    batch_names = get_files_with_extension(path, DATASET_EXT)
    for name in batch_names:
        data, label = extract_single_batch(path, name)

def extract_single_batch(path, name):
    with open(path+name, "rb") as f:
        batch = pickle.load(f)
    data = batch['data']
    label = batch['label']
    return data, label

