import math
import numpy as np
from time import sleep

def sigmoid(x):
    x = np.clip(x,-600,600)
    try:
        return 1/(1+np.exp(-x))
    except OverflowError:
        if(x < 0):
            return 0
        else:
            return 1
        
def dsigmoid(x):
    x = np.clip(x,-600,600)
    return sigmoid(x)*(1-sigmoid(x))
        
def tanh(x):
    x = np.clip(x,-600,600)
    return np.tanh(x)

def tanh_prime(x):
    x = np.clip(x,-600,600)
    return 1-(tanh(x)**2)

def softmax(X : np.array.__class__) -> np.array.__class__:
    somme = 0
    result = []

    X = np.clip(X,-600, 600)
    for x in X:
        try:
            somme += math.exp(x)
        except OverflowError:
            print(x)
            exit(1)
    for x in X:
        result.append(math.exp(x)/somme)
    return result

def dsoftmax(X : np.array.__class__) -> np.array.__class__:
    result = []
    for i in range(len(X)):
        result.append(X[i] * (1-X[i]))
    return np.array(result)

def relu(x):
    x = np.clip(x,-600,600)
    return np.maximum(0,x)


class Layer():
    #input_size : the number of neurones of the previous layer
    #output_size : the number of neurones in this layer
    def __init__(self,input_size : int,output_size : int,activation : callable,learningRate : float=1,parents : list=None):
        #self.nerone : list = []
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.learningRate = learningRate
        if(activation == tanh):self.activationPrime = tanh_prime
        elif(activation == sigmoid):self.activationPrime = dsigmoid
        elif(activation == relu):self.activationPrime = relu
        elif(activation == softmax):self.activationPrime = dsoftmax
        if(parents == None):
            self.W = np.random.uniform(low=-1,high=1,size=(output_size,input_size))
            self.B = np.random.uniform(low=-1,high=1,size=(output_size))
        else:
            self.W = (parents[0].W+parents[1].W)/2# * np.random.uniform(low=0.9,high=1.1,size=(output_size,input_size))
            self.B = (parents[0].B+parents[1].B)/2# * np.random.uniform(low=0.9,high=1.1,size=(output_size))
    
    #this function return the result of each of the neurones of this layer
    def forward(self,this_input : np.array.__class__) -> np.array.__class__:
        self.X = this_input
        self.Y = self.activation(np.dot(self.W,this_input)+self.B)
        return self.Y
    
    #error : an array containing the error for each of the neurones of this layer
    def backward(self,output_error : np.array.__class__):
        output_error *= self.activationPrime(self.Y)
        self.W += (output_error[:,np.newaxis]*self.X)*self.learningRate
        self.B += output_error*self.learningRate
        return np.dot(output_error,self.W*self.X)
    
class Networks():
    def __init__(self,neuroneNumber : list[int],activation=None,learningRate : float=1,neuroneActivation : list[callable]=None,parents : list=None) -> None:
        self.layers : list[Layer] = []
        activationList = []
        if(neuroneActivation == None):
            if(activation == None):
                raise Exception("No activation function was provided")
            for i in range(len(neuroneNumber)):
                activationList.append(activation)
        else:
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
