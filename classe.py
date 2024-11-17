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
    return sigmoid(x)*(1-sigmoid(x))
        
def tanh(x):
    x = np.clip(x,-600,600)
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def tanh_prime(x):
    x = np.clip(x,-600,600)
    return 1-(tanh(x)**2)

def softmax(X : list[float]) -> list[float]:
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

def relu(x):
    return max(0,x)

def drelu(x):
    if(x <= 0):return 0
    return 1


class Perceptron():#Single nerone
    def __init__(self,input_size : int,activation : callable,learningRate : float=1,parents : list=None):
        if(parents == None):
            self.W : np.array.__class__ = np.random.uniform(low=-1, high=1, size=(input_size,))#generating random connection wheights to every previous nerones
        else:
            self.W = (parents[0].W+parents[1].W)/2
            for w in self.W:
                w += random.uniform(-0.1,0.1)
        self.activation = activation
        if(activation == tanh):self.activationPrime = tanh_prime
        elif(activation == sigmoid):self.activationPrime = dsigmoid
        elif(activation == relu):self.activationPrime = drelu
        elif(activation == softmax):pass
        else:raise ValueError()
        self.learningRate = learningRate
        if(parents == None):
            self.bias = random.uniform(-1,1)
        else:
            self.bias = (parents[0].bias+parents[1].bias)/2 + random.uniform(-0.1,0.1)

    def forward(self,this_input : np.array.__class__) -> float:
        self.input : np.array.__class__ = this_input
        a = np.dot(this_input,self.W)
        if(self.activation == softmax):
            return (a+self.bias)
        result = self.activation(a + self.bias)
        self.output : float = result
        return result
    
    def backward(self,error : float):
        if(self.activation != softmax):
            error *= self.activationPrime(self.output)
        a = error*np.array(self.input)
        self.W += a*self.learningRate#adjusting the weight
        self.bias += self.learningRate*error
        return error*self.W*self.input

class Layer():
    #input_size : the number of neurones of the previous layer
    #output_size : the number of neurones in this layer
    def __init__(self,input_size : int,output_size : int,activation : callable,learningRate : float=1,parents : list=None):
        self.nerone : list = []
        self.input_size = input_size
        self.activation = activation
        if(activation == softmax):
            self.softmaxStore = []
        for i in range(output_size):
            if(parents == None):
                self.nerone.append(Perceptron(input_size,activation,learningRate))
            else:
                self.nerone.append(Perceptron(input_size,activation,learningRate,parents=[parents[0].nerone[i],parents[1].nerone[i]]))
    
    #this function return the result of each of the neurones of this layer
    def forward(self,this_input : np.array.__class__) -> np.array.__class__:
        result = []
        for i in range(len(self.nerone)):
            result.append(self.nerone[i].forward(this_input))
        if(self.activation == softmax):
            result = softmax(result)
            for i in range(len(self.nerone)):
                self.nerone[i].output = result[i]
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
    
    def forward(self,this_input) -> list[float]:
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
