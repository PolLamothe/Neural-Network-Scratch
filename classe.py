import math
import numpy as np
from time import sleep
import random

def sigmoid(x):
    try:
        return 1/(1+math.exp(-x))
    except OverflowError:
        if(x < 0):
            return 0
        else:
            return 1
        
def dsigmoid(x):
    return x*(1-x)
        
''' I haven't successfuly implentented these for the moment
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2
'''

class Perceptron():#Single nerone
    def __init__(self,input_size : int,hidden : bool,learningRate : float=1):
        self.W : np.array.__class__ = np.random.rand(1,input_size)[0]#generating random connection wheights to every previous nerones
        self.learningRate = learningRate
        self.hidden = hidden

    def forward(self,this_input : np.array.__class__) -> float:
        self.input : np.array.__class__ = this_input
        a = 0
        for i in range(len(this_input)):
            a += (this_input[i]*self.W[i])

        result = sigmoid(a)
        self.output : float = result
        return result
    
    def backward(self,error : float):
        if(self.hidden):
            error *= dsigmoid(self.output)#if the error come from a next nerones
        a = error*np.array(self.input)
        self.W = self.W+a*self.learningRate#adjusting the weight
        return error*self.W*self.input

class Layer():
    #input_size : the number of neurones of the previous layer
    #output_size : the number of neurones in this layer
    def __init__(self,input_size : int,output_size : int,hidden : bool,learningRate : float=1):
        self.nerone : list = []
        self.input_size = input_size
        for i in range(output_size):
            self.nerone.append(Perceptron(input_size,hidden,learningRate))
    
    #this function return the result of each of the neurones of this layer
    def forward(self,this_input : np.array.__class__) -> np.array.__class__:
        result = []
        for i in range(len(self.nerone)):
            result.append(self.nerone[i].forward(this_input))
        return np.array(result)
    
    #error : an array containing the error for each of the neurones of this layer
    def backward(self,output_error : np.array.__class__):
        nerones_error = []
        nerones_weight = []
        nerones_output = []
        for i in range(len(self.nerone)):#pour chaque neurone
            nerones_error.append(self.nerone[i].backward(output_error[i]))
            nerones_weight.append(self.nerone[i].W)
            nerones_output.append(self.nerone[i].output)
        input_error = []
        for i in range(self.input_size):
            temp = 0
            for j in range(len(self.nerone)):
                temp += nerones_error[j][i]#*nerones_weight[j][i]
            input_error.append(temp)
        return input_error
    
class Networks():
    def __init__(self,neuroneNumber : list[int],learningRate : float=1) -> None:
        self.layers = []
        for i in range(1,len(neuroneNumber)):
            if(i == len(neuroneNumber)-1):
                self.layers.append(Layer(neuroneNumber[i-1],neuroneNumber[i],False,learningRate))
            else:
                self.layers.append(Layer(neuroneNumber[i-1],neuroneNumber[i],True,learningRate))
    
    def forward(self,this_input):
        first = True
        previousResult = None
        for layer in self.layers:
            if(first):
                previousResult = layer.forward(this_input)
                first = False
            else:
                previousResult = layer.forward(previousResult)
        return previousResult

    def backward(self,output_error):
        first = True
        previousResult = None
        for i in range(len(self.layers)-1,-1,-1):
            if(first):
                previousResult = np.array(self.layers[i].backward(output_error))
                first = False
            else:
                previousResult = np.array(self.layers[i].backward(previousResult))