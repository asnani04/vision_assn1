Setup Instructions
------------------

Unzip the folder

1. download and install python3 and numpy, and tensorflow for running input.py to get the MNIST dataset in proper format.

2. Open calling.py and uncomment the model construction part. The user can change the network parameters - number of hidden layers, the number of nodes in each hidden layer, and the choice of activation function (0 for tanh and 1 for relu). The call to train can be changed to use some other optimizer as well: "sgd_optimizer", "momentum_optimizer" or "adam_optimizer".

3. python3 calling.py
