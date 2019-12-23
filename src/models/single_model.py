import tensorflow as tf
from capacity.cap_estimate import *
from .toy_model import *
from tools.dataset import *

#layers = [15,12,10,8,6] # 301
#layers = [16, 15, 10, 9, 8, 7] #331
#layers = [20, 15, 10 ] # 376
layers = [50,20, 15, 10] # 432
print(cap_estimate(layers, 15, 1))

cp_callback = lambda idx : tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir(idx)+checkpoint_path,
                                                 monitor = "acc",
                                                 verbose=1,
                                                 #save_best_only = True,
                                                 save_weights_only=True,
                                                 mode = "auto",
                                                 save_freq="epoch")
tb_callback = lambda idx : tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path(idx),
                                                 update_freq = 100)


full_dataset = input_fn(str(default_path / "3ops/alu_6.csv"), 15, [True for i in range(1)] + [False for i in range(5)]).repeat()
train_on(layers, 15, 1, full_dataset, None, [cp_callback_no_valid, tb_callback_no_valid], 3)
