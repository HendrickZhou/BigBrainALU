import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

class LogicOps:
    def __init__(self, ops, nbits, pretrained):
        """
        ops: 1 for and , 2 for or, 3 for xor
        nbits: bits of ops
        pretrained: use the weight pre-calculated, the net would be untrainable
        """
        models = [self.n_AND, self.n_OR, self.n_XOR]
        if ops is not in [1,2,3]:
            raise Exception("Illegal ops type, use 1,2,3")
        self.model = models[ops+1]
        self.bits = nbits
        self.pretrained = pretrained
        self._init_train_params()

    def __call__(self, datasets, *args):
        """
        datasets is dict of training and validation set
        """
        if "train" not in datasets.keys() or "valid" not in datasets.keys():
            raise Exception("Plz use dictionary as dataset, and include train and valid set both")
        self.train_set = datasets["train"]
        self.valid_set = datasets["valid"]

        #for _ in epoches:
        #    self.train_step()
        #    self.valid_step()

        
    
    def _init_train_params(self):
        self.optimizer = Adam()
        self.loss = BinaryCrossentropy()
        # epochs
        # test steps
        # checkpoints
        # summary of graph

    @property
    def optimizer(self):
        return self.optimizer

    @property.setter
    def optimizer(self, optimizer):
        """
        you need to set the params of optimizer yourself
        """
        self.optimizer = optimizer

    @property
    def loss(self):
        return self.loss
    
    @propety.setter
    def loss(self, loss_fn):
        """
        accept the keras loss func, or custom func
        """
        self.loss = loss_fn
    
    @tf.function
    def train_step(self):
        #for x, y in self.train_st
        pass

    @tf.function
    def valid_step(self):
        pass

