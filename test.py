from ALU import *
import pickle
import numpy as np

batch_name = lambda idx : "ALU-8-14_batch_" + str(idx) + ".batch"
folder_path = "/Users/zhouhang/Project/ALU-dataset/dataset/"

def read_patch(path, filename):
    with open(path + filename, 'rb') as f:
        batch=pickle.load(f)
    oprs = batch['operations']
    data = batch['data']
    label = batch['label']
    return data, label, oprs

def read_patch_folder(path, n):
    data, label, oprs = read_patch(path, batch_name(n))
    print(data.dtype)
    print(label.shape)
    print(oprs)
    return data, label

i = 1
data, label = read_patch_folder(folder_path, i)
