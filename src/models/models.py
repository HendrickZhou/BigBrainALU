from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import numpy as np
from tools.utils import *
from tools.dataset import *
from models.model_factory.base_model import NBitsClassifier

BITS = 6
OPS_BITS = 4


class ALUModel(NBitsClassifier):
	def __init__(self, bits):
		self.super(self, bits)

	def 