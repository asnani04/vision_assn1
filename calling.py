import numpy as np
import matplotlib.pyplot as plt
import test
import mlp

input_size = 784
output_size = 10
num_epochs = 10
batch_size = 64

class stats(object):
    def __init__(self):
        self.acc = []
        self.train_loss = []
        self.valid_loss = []
        self.epochs = []


        
train_data, train_labels, test_data, test_labels, validation_data, validation_labels = test.get_data()

shape = train_data.shape
train_data = np.reshape(train_data, (shape[0], shape[1]*shape[2]))
validation_data = np.reshape(validation_data, (validation_data.shape[0], shape[1]*shape[2]))
test_data = np.reshape(test_data, (test_data.shape[0], shape[1]*shape[2]))



model = mlp.Multi_layer_perceptron(0, [], 1)
f1 = open("h0_adam.txt", "w")

for epoch in range(num_epochs):
    acc, loss = model.train(train_data[:50000], train_labels[:50000], validation_data[:5000], validation_labels[:5000], "adam_minibatch")
    print(acc, loss)
    f1.write("%d, %f, %f\n" % (epoch+1, acc, loss))
    
f1.close()

del model

model = mlp.Multi_layer_perceptron(3, [100, 50, 25], 1)
f1 = open("h2_100_50_25_adam.txt", "w")

for epoch in range(num_epochs):
    acc, loss = model.train(train_data[:50000], train_labels[:50000], validation_data[:5000], validation_labels[:5000], "adam_minibatch")
    print(acc, loss)
    f1.write("%d, %f, %f\n" % (epoch+1, acc, loss))
    
f1.close()

del model


