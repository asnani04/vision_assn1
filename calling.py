import numpy as np
#import matplotlib.pyplot as plt
import inputs as test
import mlp

input_size = 784
output_size = 10
num_epochs = 10
batch_size = 64


        
train_data, train_labels, test_data, test_labels, validation_data, validation_labels = test.get_data()

shape = train_data.shape
train_data = np.reshape(train_data, (shape[0], shape[1]*shape[2]))
validation_data = np.reshape(validation_data, (validation_data.shape[0], shape[1]*shape[2]))
test_data = np.reshape(test_data, (test_data.shape[0], shape[1]*shape[2]))




model = mlp.Multi_layer_perceptron(1, [25], 1)
f1 = open("h1_25_adam_2.txt", "w")

for epoch in range(num_epochs):
    acc, loss = model.train(train_data[:50000], train_labels[:50000], validation_data[:5000], validation_labels[:5000], "adam_minibatch")
    print(acc, loss)
    f1.write("%d, %f, %f\n" % (epoch+1, acc, loss))
    
f1.close()

del model

# model = mlp.Multi_layer_perceptron(2, [100, 25], 1)
# num_grads = model.numerical_gradients(train_data[:1], train_labels[:1])


