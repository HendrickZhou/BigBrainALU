import pandas as pd
import numpy as np
import random
from tools.utils import default_dataset_path

default_path = default_dataset_path()

# use system time as the seed
random.seed()

def random_tag(csv_filepath, csv_name, new_name, ratio=0.7):
    """
    ratio: of the training set
    """
    #don't use pandas iterator as it's inefficient and couterintuitive
    df = pd.read_csv(str(csv_filepath / csv_name))
    train_test = np.zeros(len(df.index), np.bool)
    for i in range(len(df.index)):
        if random.random() <= ratio:
            train_test[i] = True 
    
    df["set"] = train_test
   
    #train_set = df[df["set"] == True]
    #valid_set = df[df["set"] == False]
    train_set.to_csv(str(csv_filepath / (new_name+"_train.csv")), header=False, index=False)
    valid_set.to_csv(str(csv_filepath / (new_name+"_valid.csv")), header=False, index=False)
#    df.to_csv(str(csv_filepath / new_name), header = False, index = False)
#    print(df)
#    train = len(df[df["set"] == True])
#    test = len(df[df["set"] == False])
#    print(train)
#    print(test)

def measure_conv(csv_filepath, csv_name):
    """
    measure if the dataset division converges
    """
    pass

if __name__ == "__main__":
    random_tag(default_path/"3ops", "alu_6.csv", "alu_6", 0.85)
