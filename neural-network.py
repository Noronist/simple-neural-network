import numpy as np
import random

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_inputs, num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_hidden, num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer(hidden_layer_weights)
        self.init_weights_from_hidden_layer_to_output_layer(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer(self, hidden_layer_weights):
        self.hidden_layer.weights = hidden_layer_weights

    def init_weights_from_hidden_layer_to_output_layer(self, output_layer_weights):
        self.output_layer.weights = output_layer_weights

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return hidden_layer_outputs, self.output_layer.feed_forward(hidden_layer_outputs)


    def train(self, training_inputs, training_outputs):
        hidden_layer_outputs, output_layer_outputs = self.feed_forward(training_inputs) #(out_h(1,2))

        pd_errors_wrt_output_neuron_total_net_input = self.output_layer.calculate_pd_error_wrt_total_net_input(training_outputs)
        
        pd_total_net_input_wrt_input = self.hidden_layer.calculate_pd_total_net_input_wrt_input()

        pd_error_wrt_hidden_layer_weight_matrix = pd_errors_wrt_output_neuron_total_net_input * self.hidden_layer.output
        
        #New output neurons' weights
        new_output_layer_weights = self.output_layer.weights - self.LEARNING_RATE*np.tile((pd_error_wrt_hidden_layer_weight_matrix),(2,1))
        #---------------------------#
        
        pd_total_wrt_output_of_hidden_layer = np.sum(pd_errors_wrt_output_neuron_total_net_input*self.output_layer.weights,axis=1)
        der_ = pd_total_wrt_output_of_hidden_layer*pd_total_net_input_wrt_input*np.tile(training_inputs.T,(1,2))
        
        #New hidden neurons' weights
        new_hidden_layer_weights = self.hidden_layer.weights - self.LEARNING_RATE*der_
        
        #Update neurons' weights
        self.output_layer.weights = new_output_layer_weights
        self.hidden_layer.weights = new_hidden_layer_weights

    def calculate_total_error(self, training_sets):
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            training_inputs = np.array([training_inputs])
            training_outputs = np.array([training_outputs])
            
            self.feed_forward(training_inputs)
            total_error = self.output_layer.calculate_error(training_outputs)  
            return total_error

class NeuronLayer:
    def __init__(self, num_inputs, num_neurons, bias):
        self.bias = bias
        self.weights = np.empty([num_inputs, num_neurons])

    def feed_forward(self, inputs):
        self.output =  self.calculate_output(inputs)
        return self.output
    
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output
        
    def squash(self, total_net_input):
        return 1 / (1 + np.exp(-total_net_input))
    
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();
    
    def calculate_total_net_input(self):
        total = np.matmul(self.inputs, self.weights)
        return total + self.bias

    def calculate_error(self, target_output):
        return np.sum(0.5 * (target_output - self.output)** 2, axis=1)

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)


# Blog post example:
input_matrix = np.array([[0.05, 0.1]])
target_matrix = np.array([[0.01, 0.99]])
hidden_layer_weights_matrix = np.array([[0.15, 0.25],[0.2, 0.3]])
output_layer_weights_matrix = np.array([[0.4, 0.5],[0.45, 0.55]])
hidden_layer_bias_matrix = np.array([[0.35,0.35]])
output_layer_bias_matrix = np.array([[0.6, 0.6]])
nn = NeuralNetwork(2, 2, 2, hidden_layer_weights_matrix, hidden_layer_bias_matrix, output_layer_weights_matrix, output_layer_bias_matrix)
#nn.train(input_matrix, target_matrix
#print((np.round_(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9)))

for i in range(1000):
    nn.train(input_matrix, target_matrix)
    print(i, (np.round_(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9)))
    
print(nn.output_layer.output)

# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))
