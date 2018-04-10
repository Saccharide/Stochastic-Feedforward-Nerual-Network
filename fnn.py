# This is the example as laid out by "Make Your Own Neural Network" by Tariq Rashid
# This will be the first step in created our Stochastic Feedforward Neural Network
# Author: Cody Watson

# import the numpy environment for python
# import the scipy environment for python
import numpy as np
import scipy as sp

# Neural Network class definition
class neuralNetwork:
    
    # initialize neural net
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        # attributes of the neural network
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # learning rate for neural net
        self.lr = learningrate
        
        # weight matrices for input to hidden and hidden to output
        # these weights are taken from a normal probability distribution
        # from 1 to -1.
        self.wih = np.random.normal(0.0, pow(self.hnodes, -1), (self.hnodes, self.inodes))
        print(self.wih)
        self.who = np.random.normal(0.0, pow(self.onodes, -1), (self.onodes, self.hnodes))
        
        # creation of the activation function
        self.activation_function = lambda x: sp.special.expit(x)
        
        pass
        
        
    # function to train the neural net
    def train(self, inputs_list, targets_list):
        
        # convert inputs and targets into a 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # what comes into the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # what comes out of the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # what comes into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate final answer from output layer
        final_outputs = self.activation_function(final_inputs)
        
        # begin to calculate erros and backpropagate
        output_errors = targets - final_outputs
        
        # calculates the error of the hidden layer
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # after calculating errors, begin updating weights
        # begin with updating weights between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # updating the weights between hidden and input layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    
        pass
        
    # function to query the neural network
    def query(self, inputs_list):
        
        # converting inputs into a 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # what comes into the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # what comes out of the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # what comes into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate final answer from output layer
        final_outputs = self.activation_function(final_inputs)
        
        final = np.random.binomial(1, final_outputs)
        print(final)
        
        return final_outputs
        
        
    
