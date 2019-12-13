#!/usr/bin/env python

from ALU import *
import numpy as np
import pandas as pd
import pickle

class Dataset():
    def __init__(self, data_bits, path, label_bit_msk=None):
        if label_bit_msk is None:
            label_bit_msk = [True for _ in range(data_bits)]
        elif(len(label_bits_msk) >= data_bits):
            raise Exception("unsupported label bit mask length")
        self.path = path
        self.data_bits = data_bits
        self.label_bit_msk = [i!=0 for i in label_bit_msk]
        
        self.alu = ALU(self.data_bits)
        self.data_dim = self.alu.data_dim
        self.label_dim = min(self.alu.label_dim, len(self.label_bit_msk))
        self.filename = str()

    def __iter__(self):
        """
        only support generating the whole table now
        """
        number, ops = self.alu.gen_range()
        arr = lambda x : np.array(x, dtype = "uint8")
        for op in ops:
            for B in number:
                for A in number:
                    data, label = self._get_data_label(A, B, op)
                    yield arr(data), arr(label)

    def __call__(self, form = "csv", batch_size = 1000):
        if form is "csv":
            self.path = self.path + "dataset_csv/"
            self.filename = "alu_{}.csv".format(self.data_bits)
            self._csv()

        elif form is "batch":
            self.path = self.path + "dataset{}".format(self.data_bits)
            number, ops = self.alu.gen_range()
            datas = []
            labels = []
            operations = []
            data_dim = self.data_dim
            label_dim = self.label_dim
            total_size = len(ops) * len(number)**2
            i = 0
            for op in ops:
                for B in number:
                    for A in number:
                        data, label = self._get_data_label(A, B, op)

                        datas.append(data)
                        labels.append(label)
                        operations.append(op)
                        i = i + 1
                        if i%batch_size is 0 or i is total_size:
                            name = self.filename + "_"+ str(i//batch_size)
                            actual_size = batch_size if i % batch_size is 0 else i % batch_size
                            data_arr = np.array(datas, dtype= 'uint8').reshape((actual_size, data_dim))
                            label_arr = np.array(labels, dtype = 'uint8').reshape((actual_size, label_dim))
                            dataset = dict()
                            dataset["data"] = data_arr
                            dataset["label"] = label_arr
                            dataset["operations"] = operations
                            with open(self.path + name + '.batch', 'wb+') as f:
                                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                            datas = []
                            labels = []
                            operations = []
        else:
            raise Exception("Illegal format type")
         

    def _csv(self):
        number, ops = self.alu.gen_range()
        datas = []
        labels = []
        data_dim = self.alu.data_dim
        label_dim = self.alu.label_dim
        total_size = len(ops) * len(number)**2
        i = 0

        
        for op in ops:
            for B in number:
                for A in number:
                    data, label = self._get_data_label(A, B, op)
                    datas.append(data)
                    labels.append(label)


        data_arr = np.array(datas, dtype='uint8').reshape((total_size, data_dim))
        label_arr = np.array(labels, dtype = 'uint8').reshape((total_size, label_dim))
        df = pd.DataFrame(np.hstack((data_arr, label_arr)))
        df.to_csv(self.path + self.filename, header=False, index=False)

    def _get_data_label(self, A, B, op):
        """
        return the list of data and label
        """
        in1, in2, opc, out = self.alu(A, B, op)
        data = list(in1) + list(in2) + list(opc)
        label = list(out)
        label = [i for i,j in zip(label, self.label_bit_msk) if j]
        return data, label  



if __name__ == '__main__':
    import os
    script_path = os.path.abspath(__file__)
    project_dir = script_path[:script_path.rfind("src")]
    output_path = project_dir + "dataset/"
    # ds = Dataset(6, "ALU-6-14_batch", output_path)
    ds = Dataset(8, output_path)
    ds()
    # for data, label in iter(ds):
    #     print(data)
    #     print(label)
