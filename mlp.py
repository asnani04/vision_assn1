import numpy as np
import inputs as test

input_size = 784
output_size = 10

class Multi_layer_perceptron(object):
    def __init__(self, no_hidden, node_vector, act_fun):
        '''
        params:
        no_hidden: int, number of hidden layers desired
        node_vector: [1 x no_hidden]: vector of ints for number of nodes in each layer
        act_fun: int, activation function (common for all layers for now)
        '''

        self.no_hidden = no_hidden
        self.node_vector = node_vector
        self.act_fun = act_fun
        self.node_vector = [input_size] + self.node_vector + [output_size]
        print(self.node_vector)
        self.loss = 0.0
        #for now, assume just one hidden layer

        self.params, self.activations, self.gradients, self.node_grads = {}, {}, {}, {}
        self.prev_update = {}
        self.sq_gradients = {}
        self.avg_grad = {}
        self.avg_squared_grad = {}
        
        for i in range(no_hidden + 1):
            wt_string = "weights" + str(i)
            act_string = "acts" + str(i)
            node_string = "node" + str(i)
            weights = np.random.normal(0, 0.1, (self.node_vector[i], self.node_vector[i+1]))
            grads = np.zeros_like(weights)
            sq_grads = np.zeros_like(weights)
            prev_update = np.zeros_like(weights)
            avg_grad = np.zeros_like(weights)
            avg_sq_grad = np.zeros_like(weights)
            acts = np.zeros(self.node_vector[i])
            node_grads = np.zeros(self.node_vector[i])
            
            self.params[wt_string] = weights
            self.gradients[wt_string] = grads
            self.sq_gradients[wt_string] = sq_grads
            self.prev_update[wt_string] = prev_update
            self.avg_grad[wt_string] = avg_grad
            self.avg_squared_grad[wt_string] = avg_sq_grad
            self.activations[act_string] = acts
            self.node_grads[node_string] = node_grads
            
        acts = np.zeros(self.node_vector[-1])
        node_grads = np.zeros(self.node_vector[-1])
        act_string = "acts" + str(no_hidden+1)
        self.activations[act_string] = acts
        self.node_grads["node" + str(no_hidden+1)] = node_grads

        self.running_beta1 = None
        self.running_beta2 = None
        
    def nonlinear(self, inputs):
        '''
        Computes nonlinearity in a 1D inputs array
        '''
        
        if self.act_fun == 0:
            return tanh(inputs)
        elif self.act_fun == 1:
            return relu(inputs)


    def forward_pass(self, inputs):
        '''
        Computes a forward pass on the inputs and stores the node activations at each node
        '''
        
        self.activations["acts0"] = inputs
        acts = inputs
        for i in range(self.no_hidden + 1):
            acts = np.matmul(acts, self.params["weights" + str(i)])
            acts = self.nonlinear(acts)
            self.activations["acts" + str(i+1)] = acts
        
        self.outputs = self.activations["acts" + str(self.no_hidden + 1)]
        return self.outputs

    def softmax(self):
        '''
        Computes softmax probabilities of output labels
        '''
        self.probs = np.zeros_like(self.outputs)
        normalizer = sum(np.exp(self.outputs))
        for i in range(output_size):
            self.probs[i] = float(np.exp(self.outputs[i]) / normalizer)
        return self.probs

    def compute_loss(self, true_label):
        '''
        Computes cross entropy loss with softmax probabilities
        '''
        self.softmax_loss = np.zeros_like(self.probs)
        for i in range(output_size):
            self.softmax_loss[i] = true_label[i]*np.log(self.probs[i] + 1e-8)
        self.loss += (-1) * sum(self.softmax_loss)
        return self.loss
    
    def compute_gradients(self, true_label, batch_size=64):
        '''
        Computes gradients of the loss with respect to all the parameters
        '''
        self.node_grads["node" + str(self.no_hidden + 1)] = np.subtract(self.probs, true_label)
        # print(self.node_grads["node" + str(self.no_hidden + 1)].shape)
        for i in range(self.no_hidden, 0, -1):
            node_grads = np.matmul(self.params["weights" + str(i)], self.node_grads["node" + str(i+1)])
            # print(node_grads.shape)
            if self.act_fun == 0:
                layer_grad = grad_tanh(self.activations["acts" + str(i)])
            elif self.act_fun == 1:
                layer_grad = grad_relu(self.activations["acts" + str(i)])
            node_grads = np.multiply(node_grads, layer_grad)
            self.node_grads["node" + str(i)] = node_grads
            # print(self.node_grads["node" + str(i)].shape)

        for i in range(self.no_hidden + 1):
            if self.act_fun == 0:
                layer_grad = grad_tanh(self.activations["acts" + str(i+1)])
            elif self.act_fun == 1:
                layer_grad = grad_relu(self.activations["acts" + str(i+1)])
            weight_grads = np.outer(self.activations["acts" + str(i)], np.multiply(layer_grad, self.node_grads["node" + str(i+1)]))
      
            # for l in range(self.gradients["weights" + str(i)].shape[0]):
            #     for b in range(self.gradients["weights" + str(i)].shape[1]):
            #         grad = self.gradients["weights" + str(i)][l][b]
            #         if grad + weight_grads[l][b] < batch_size * 5:
            #             self.gradients["weights" + str(i)][l][b] = grad + weight_grads[l][b]
            self.gradients["weights" + str(i)] = np.add(self.gradients["weights" + str(i)], weight_grads)
            upper_cap = batch_size * 5 * np.ones_like(self.gradients["weights" + str(i)])
            self.gradients["weights" + str(i)] = np.minimum(self.gradients["weights" + str(i)], upper_cap)
            self.sq_gradients["weights" + str(i)] = np.square(self.gradients["weights" + str(i)])
            
        return self.gradients

    def apply_gd_optimizer(self, learning_rate, div_factor):
        '''
        Applies gradient descent optimizer to the params
        '''
        for i in range(self.no_hidden + 1):
            self.params["weights" + str(i)] = np.subtract(self.params["weights" + str(i)], ((learning_rate / div_factor) * self.gradients["weights" + str(i)]))
            self.gradients["weights" + str(i)] = np.zeros_like(self.gradients["weights" + str(i)])
            
    def apply_momentum_optimizer(self, div_factor, learning_rate = 0.001,  momentum = 0.009):
        '''
        Applies gradient descent optimizer with momentum to the params
        momentum: coefficient to use with previous momentum
        '''
        for i in range(self.no_hidden + 1):
            self.prev_update["weights" + str(i)] = np.add((momentum * self.prev_update["weights" + str(i)]), (learning_rate / div_factor) * self.gradients["weights" + str(i)])
            self.params["weights" + str(i)] = np.subtract(self.params["weights" + str(i)], self.prev_update["weights" + str(i)])
            self.gradients["weights" + str(i)] = np.zeros_like(self.gradients["weights" + str(i)])

    def apply_adam_optimizer(self, learning_rate, div_factor, beta1 = 0.9, beta2 = 0.999):
        '''
        Applies Adam optimizer to the params
        '''
        if self.running_beta2 is None:
            self.running_beta2 = beta2
        else:
            self.running_beta2 = self.running_beta2 * beta2

        if self.running_beta1 is None:
            self.running_beta1 = beta1
        else:
            self.running_beta1 = self.running_beta1 * beta1
            
        for i in range(self.no_hidden + 1):
            self.avg_grad["weights" + str(i)] = np.add(beta1 * self.avg_grad["weights" + str(i)], ((1 - beta1) / div_factor) * self.gradients["weights" + str(i)])
            self.avg_squared_grad["weights" + str(i)] = np.add(beta2 * self.avg_squared_grad["weights" + str(i)], ((1 - beta2) / div_factor) * self.sq_gradients["weights" + str(i)])
            self.unbiased_avg_grad = self.avg_grad["weights" + str(i)] / (1 - self.running_beta1)
            self.unbiased_avg_sq_grad = self.avg_squared_grad["weights" + str(i)] / (1 - self.running_beta2)
            epsilon = 1e-8
            for l in range(self.params["weights" + str(i)].shape[0]):
                for b in range(self.params["weights" + str(i)].shape[1]):
                    self.params["weights" + str(i)][l][b] -= learning_rate * self.unbiased_avg_grad[l][b] / (epsilon + np.sqrt(self.unbiased_avg_sq_grad[l][b]))
            self.gradients["weights" + str(i)] = np.zeros_like(self.gradients["weights" + str(i)])
            self.sq_gradients["weights" + str(i)] = np.zeros_like(self.sq_gradients["weights" + str(i)])
                    
    def prediction(self):
        pred = np.argmax(self.probs)
        return pred
    
    def train(self, data, labels, valid_data, valid_labels, optimizer, batch_size = 64):
        '''
        Trains the model on the data for one epoch
        '''
        if optimizer == "gd_batch":
            '''
            Use batch gradient descent. Slow to train.
            '''
            for i in range(data.shape[0]):
                inputs = data[i]
                outputs = self.forward_pass(inputs)
                probs = self.softmax()
                loss = self.compute_loss(labels[i])
                grads = self.compute_gradients(labels[i], data.shape[0])
                # print(self.gradients["weights0"][1][1])
            self.apply_gd_optimizer(0.1, data.shape[0])
            self.loss = 0.0

        else:
            train_size = data.shape[0]
            no_batches = train_size // batch_size
            for i in range(no_batches):
                mini_batch_indices = np.random.permutation(train_size)[:batch_size]
                mini_batch = []
                for j in range(batch_size):
                    inputs = data[mini_batch_indices[j]]
                    outputs = self.forward_pass(inputs)
                    probs = self.softmax()
                    loss = self.compute_loss(labels[mini_batch_indices[j]])
                    grads = self.compute_gradients(labels[mini_batch_indices[j]], batch_size)
                    # print(self.gradients["weights0"][1][1])
                if optimizer == "sgd_minibatch":
                    '''
                    Use stochastic mini batch gradient descent.
                    '''
                    self.apply_gd_optimizer(0.001, batch_size)

                if optimizer == "momentum_minibatch":
                    '''
                    Use mini batch gradient descent with momentum
                    '''
                    self.apply_momentum_optimizer(batch_size, 0.001)
                if optimizer == "adam_minibatch":
                    '''
                    Use mini batch adam optimizer
                    '''
                    self.apply_adam_optimizer(0.001, batch_size)
            
        self.loss = 0.0
        correct = 0
        
        for i in range(valid_data.shape[0]):
            outputs = self.forward_pass(valid_data[i])
            probs = self.softmax()
            prediction = self.prediction()
            if prediction == np.argmax(valid_labels[i]):
                correct += 1

        accuracy = float(correct / valid_data.shape[0])
        return accuracy, loss / data.shape[0]

    def numerical_gradients(self, data, labels):
        perturbation = 0.0001
        old_loss = 0.0
        self.num_gradients = {}
        for i in range(data.shape[0]):
            inputs = data[i]
            outputs = self.forward_pass(inputs)
            probs = self.softmax()
            old_loss = old_loss + self.compute_loss(labels[i])
            # print("old loss = %f" % old_loss)
            self.model_grads = self.compute_gradients(labels[i], data.shape[0])

        self.loss = 0.0
        # for j in range(self.no_hidden + 1):
        #     self.gradients["weights" + str(i)] = np.zeros_like(self.gradients["weights" + str(i)])
        f1 = open("num_grads.txt", "w")
        f2 = open("var_num_grads.txt", "w")
        
        layer_loss = 0.0
        
        for j in range(self.no_hidden + 1):
            self.difference = 0.0
            self.sq_difference = 0.0
            wt_string = "weights" + str(j)
            self.num_gradients[wt_string] = np.zeros_like(self.gradients[wt_string])
            for l in range(self.gradients[wt_string].shape[0]):
                for b in range(self.gradients[wt_string].shape[1]):
                    old_weight = self.params[wt_string][l][b]
                    self.params[wt_string][l][b] = old_weight + perturbation
                    new_loss = 0.0
                    for i in range(data.shape[0]):
                        inputs = data[i]
                        outputs = self.forward_pass(inputs)
                        probs = self.softmax()
                        new_loss = new_loss + self.compute_loss(labels[i])
                    delta_loss = new_loss - old_loss
                    # print("new loss = %f"% new_loss)
                    self.num_gradients[wt_string][l][b] = delta_loss / perturbation
                    # f1.write("%f\n" % (self.num_gradients[wt_string][l][b] - self.model_grads[wt_string][l][b]))
                    # print(self.num_gradients[wt_string][l][b] - self.model_grads[wt_string][l][b])
                    self.difference = self.difference + (self.num_gradients[wt_string][l][b] - self.model_grads[wt_string][l][b])
                    self.sq_difference = self.sq_difference + (self.num_gradients[wt_string][l][b] - self.model_grads[wt_string][l][b]) ** 2
                    self.params[wt_string][l][b] = old_weight
                    self.loss = 0.0
            f1.write("%d, %f\n" % (j, self.difference / (self.node_vector[i] * self.node_vector[i+1])))
            f2.write("%d, %f\n" % (j, self.sq_difference / (self.node_vector[i] * self.node_vector[i+1])))
        f1.close()            
        return self.num_gradients
                        

def relu(inputs):
    #verified forward
    '''
    Implements Relu - 6
    '''
    zero_array = np.zeros_like(inputs)
    relu_zero = np.maximum(zero_array, inputs)
    six_array = 6 * np.ones_like(inputs)
    relu_six = np.minimum(six_array, inputs)
    return relu_six

def tanh(inputs):
    #verified forward
    tanh_value = np.zeros_like(inputs)
    for i in range(len(inputs)):
        pos = np.exp(inputs[i]) if inputs[i] < 10 else 0
        neg = np.exp((-1) * inputs[i]) if inputs[i] > -10 else 0
        tanh_value[i] = ((pos - neg) / (pos + neg)) if pos+neg !=0 else 0 
    return tanh_value

def grad_tanh(inputs):
    squared_inputs = np.square(inputs)
    ones = np.ones_like(inputs)
    grads = np.subtract(ones, squared_inputs)
    return grads

def grad_relu(inputs):
    grads = np.ones_like(inputs)
    for i in range(len(inputs)):
        if inputs[i] > 0 and inputs[i] < 6:
            grads[i] = 1
        else:
            grads[i] = 0
    return grads



# model = Multi_layer_perceptron(1, [25], 1)


# train_data, train_labels, test_data, test_labels, validation_data, validation_labels = test.get_data()

# shape = train_data.shape

# train_data = np.reshape(train_data, (shape[0], shape[1]*shape[2]))
# validation_data = np.reshape(validation_data, (validation_data.shape[0], shape[1]*shape[2]))
# test_data = np.reshape(test_data, (test_data.shape[0], shape[1]*shape[2]))


# for epoch in range(num_epochs):
#     acc, loss = model.train(train_data[:50000], train_labels[:50000], validation_data[:5000], validation_labels[:5000], "momentum_minibatch")
#     print(acc, loss)

# num_grads = model.numerical_gradients(train_data[:1], train_labels[:1])
