# BigBrainALU
experimental research project on capacity theory of neural network

Mimicing bit flipping as a casual model, learning ALU caculation.

Experiment on the capacity reduction and generalization ability.

## Dataset
pipelining the dataset using tf.data API, Reading the data from sets fo files directly 


## Model building
toy models' training and saving result  
Standardized models' building process, gonna use dense network with decreasing size

Custom models' building process  
resnet?
piNN training  


Small task traning
3 bits dataset with ADD, SUB operations

## All Experiments
1. Memorization
Use all dataset as training set, test curve of memorization(this is more like an explorational experiment)

2. Generalization
Split the dataset into 2 halves and measure the generalization metrics.  
Spliting the training set will inevitably cause the incompleteness of the definition of some rules?

3. Bit hacking
Manually calcualte minimal possible bits of different logical and arithematical operations.
See if we can analyse the similar capacity of similar research: Nerual ALU/ Nerual GPU.

4. Casual Model
3 bits net with only ADD, SUB, AND, OR, XOR etc.  
check if a manual net would work  
mess up the weight, check if training helps .
