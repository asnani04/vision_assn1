Setup Instructions
------------------

Unzip the folder

1. download and install python3 and numpy, and tensorflow for running inputs.py to get the MNIST dataset in proper format.

2. Open calling.py and uncomment the model construction part. The user can change the network parameters - number of hidden layers, the number of nodes in each hidden layer, and the choice of activation function (0 for tanh and 1 for relu). The call to train can be changed to use some other optimizer as well: "sgd_optimizer", "momentum_optimizer" or "adam_optimizer".

3. python3 calling.py

4. To change other hyperparameters, open calling.py and alter the batch size, number of epochs as you wish (initialized just after the imports). 
Learning rates can be changed by opening mlp.py and altering the calls to the various optimizers inside Multi_layer_perceptron.train(). 

-----------------------



readme.pdf contains detailed description of all the classes and functions used, and the graphs and findings obtained.
