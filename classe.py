import math
import numpy as np
from time import sleep

class ActivationFunction():
    def function(X):
        raise NotImplemented("")
    
    def derivative(X):
        raise NotImplemented("")
    
class Sigmoid(ActivationFunction):
    def function(X):
        X = np.clip(X,-600,600)
        return 1/(1+np.exp(-X))
    
    def derivative(X):
        X = np.clip(X,-600,600)
        return Sigmoid.function(X)*(1-Sigmoid.function(X))

class Softmax(ActivationFunction):
    def function(X):
        X = np.clip(X,-600,600)
        exps = np.exp(X)
        return exps / np.sum(exps)
    
    def derivative(X):
        return X
    
class Tanh(ActivationFunction):
    def function(X):
        X = np.clip(X,-600,600)
        return np.tanh(X)
    
    def derivative(X):
        X = np.clip(X,-600,600)
        return 1-(Tanh.function(X)**2)
    
class NN():
    def forward():
        pass

    def backward():
        pass

class Layer():
    #input_size : the number of neurones of the previous layer
    #output_size : the number of neurones in this layer
    def __init__(self,input_size : int,output_size : int,activation : ActivationFunction,learningRate : float=1,parents : list=None):
        #self.nerone : list = []
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.learningRate = learningRate
        if(parents == None):
            self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / (input_size + output_size))
            self.B = np.random.uniform(low=-1,high=1,size=(output_size))
        else:
            self.W = (parents[0].W+parents[1].W)/2# * np.random.uniform(low=0.9,high=1.1,size=(output_size,input_size))
            self.B = (parents[0].B+parents[1].B)/2# * np.random.uniform(low=0.9,high=1.1,size=(output_size))
    
    #this function return the result of each of the neurones of this layer
    def forward(self,this_input : np.array.__class__) -> np.array.__class__:
        self.X = this_input
        self.Y = self.activation.function(np.dot(self.W,this_input)+self.B)
        return self.Y
    
    #error : an array containing the error for each of the neurones of this layer
    def backward(self,output_error : np.array.__class__):
        output_error *= self.activation.derivative(self.Y)
    
        batch_size = self.X.shape[0] if len(self.X.shape) > 1 else 1
        self.W += (output_error[:, np.newaxis] * self.X) * (self.learningRate / batch_size)
        self.B += output_error * (self.learningRate / batch_size)
        
        return np.dot(output_error, self.W)
    
class FNN(NN):
    def __init__(self,neuroneNumber : list[int],learningRate : float=1,neuroneActivation : list[ActivationFunction]=None,parents : list=None) -> None:
        self.layers : list[Layer] = []
        activationList = []
        for i in range(len(neuroneActivation)):
            activationList.append(neuroneActivation[i])
        for i in range(1,len(neuroneNumber)):
            try:
                if(parents != None):
                    self.layers.append(Layer(neuroneNumber[i-1],neuroneNumber[i],activationList[i-1],learningRate,parents=[parents[0].layers[i-1],parents[1].layers[i-1]]))
                else:
                    self.layers.append(Layer(neuroneNumber[i-1],neuroneNumber[i],activationList[i-1],learningRate))
            except IndexError:
                raise Exception("You forgot to provide the activation function for a layer")
    
    def forward(self,this_input : np.array.__class__) -> list[float]:
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