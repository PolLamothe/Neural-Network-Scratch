import math
import numpy as np
import scipy.signal
import copy
import skimage

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
    def forward():
        pass

    def backward():
        pass

class FullyConnectedLayer(Layer):
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
    def forward(self,this_input : np.ndarray) -> np.ndarray:
        self.X = this_input
        self.Y = self.activation.function(np.dot(self.W,this_input)+self.B)
        return self.Y
    
    #error : an array containing the error for each of the neurones of this layer
    def backward(self,output_error : np.ndarray):
        output_error *= self.activation.derivative(self.Y)
    
        batch_size = self.X.shape[0] if len(self.X.shape) > 1 else 1
        self.W += (output_error[:, np.newaxis] * self.X) * (self.learningRate / batch_size)
        self.B += output_error * (self.learningRate / batch_size)
        
        return np.dot(output_error, self.W)

class ConvolutionalLayer(Layer):
    def __init__(self,input_size : int,kernel_size : int,kernel_number : int,depth : int = 1,learning_rate = 1):
        self.input_size = input_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.kernel_number = kernel_number
        self.selection_size = self.input_size - self.kernel_size +1
        self.learning_rate = learning_rate

        self.K = []
        self.B = []

        for j in range(self.kernel_number):
            self.K.append([])
            self.B.append(np.random.randn(self.selection_size, self.selection_size) * np.sqrt(2. / (self.selection_size + self.selection_size)))
            for i in range(self.depth):    
                self.K[-1].append(np.random.randn(kernel_size, kernel_size) * np.sqrt(2. / (kernel_size + kernel_size)))

    def forward(self,this_input : np.ndarray) -> np.ndarray:
        result = []
        self.X = this_input

        this_output = []
        for i in range(self.kernel_number):
            somme = np.zeros((self.selection_size,self.selection_size))
            for j in range(self.depth):
                somme += scipy.signal.convolve2d(
                    this_input[j],
                    np.rot90(self.K[i][j],2)
                ,mode="valid")
            somme /= self.depth
            somme += self.B[i]
            this_output.append(copy.deepcopy(somme))
        self.Y = copy.deepcopy(this_output)
        return this_output
    
    def backward(self,error : np.ndarray) -> np.ndarray:
        for i in range(self.depth):            
            for j in range(self.kernel_number):
                self.K[j][i] += scipy.signal.convolve2d(
                    self.X[i],
                    np.rot90(error[j])
                ,mode="valid") * self.learning_rate

                self.B += error[j] * self.learning_rate

        input_error = []
        for i in range(self.depth):
            temp = None
            for j in range(self.kernel_number):
                if(temp is None):
                    temp = scipy.signal.convolve2d(
                        error[j],
                        np.rot90(self.K[j][i],2))
                else:
                    temp += scipy.signal.convolve2d(
                        error[j],
                        np.rot90(self.K[j][i],2))
            input_error.append(copy.deepcopy(temp))
        return input_error

class PoolingLayer(Layer):
    def __init__(self,input_size : int,output_size : int, max_pooling : bool = True,depth : int = 1):
        self.input_size = input_size
        self.output_size = output_size
        self.max_pooling = max_pooling
        self.depth = depth
        if(self.input_size%self.output_size != 0):
            raise Exception("The size of the layer is not valid")
        self.selection_size = self.input_size//self.output_size
    
    def forward(self,this_input : np.ndarray) -> np.ndarray:
        self.X = copy.deepcopy(this_input)
        self.Y = []
        for i in range(self.depth):
            temp = np.zeros((self.output_size,self.output_size))
            for j in range(self.input_size//self.selection_size):
                for x in range(self.input_size//self.selection_size):
                    region = this_input[i][
                        j*self.selection_size:(j+1)*self.selection_size,
                        x*self.selection_size:(x+1)*self.selection_size,
                        ]
                    if(self.max_pooling):
                        temp[j,x] = np.max(region)
                    else:
                        temp[j,x] = np.mean(region)
            self.Y.append(copy.deepcopy(temp))
        self.Y = np.array(self.Y)

        return self.Y
    
    def backward(self,error : np.ndarray) -> np.ndarray:
        result = []

        for i in range(self.depth):
            result.append(np.zeros(self.X[i].shape))
            for j in range(0,self.input_size,self.selection_size):
                for x in range(0,self.input_size,self.selection_size):
                    region = self.X[i][j:j+self.selection_size,x:x+self.selection_size]
                    if(self.max_pooling):
                        max_index = np.unravel_index(region.argmax(), region.shape)
                        result[-1][j+max_index[0]][x+max_index[1]] = error[i][j//self.selection_size,x//self.selection_size]
                    else:
                        result[-1][j:j+self.selection_size,x:x+self.selection_size] += error[i][j//self.selection_size,x//self.selection_size] / (self.selection_size**2)
        return result

class FNN(NN):
    def __init__(self,neuroneNumber : list[int],learningRate : float=1,neuroneActivation : list[ActivationFunction]=None,parents : list=None) -> None:
        self.layers : list[FullyConnectedLayer] = []
        activationList = []
        for i in range(len(neuroneActivation)):
            activationList.append(neuroneActivation[i])
        for i in range(1,len(neuroneNumber)):
            try:
                if(parents != None):
                    self.layers.append(FullyConnectedLayer(neuroneNumber[i-1],neuroneNumber[i],activationList[i-1],learningRate,parents=[parents[0].layers[i-1],parents[1].layers[i-1]]))
                else:
                    self.layers.append(FullyConnectedLayer(neuroneNumber[i-1],neuroneNumber[i],activationList[i-1],learningRate))
            except IndexError:
                raise Exception("You forgot to provide the activation function for a layer")
    
    def forward(self,this_input : np.ndarray) -> list[float]:
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

class CNN(NN):
    def __init__(self,layers : list[Layer]):
        self.layers = layers

    def forward(self,this_input : np.ndarray) -> list[float]:
        first = True
        previousResult = None
        for layer in self.layers:
            if(first):
                previousResult = layer.forward(this_input)
                first = False
            else:
                if(type(layer) == FullyConnectedLayer and previousResult.shape != (0,1)):
                    previousResult = np.append(previousResult[0],[])
                previousResult = layer.forward(previousResult)
        return previousResult

    def backward(self,output_error):
        first = True
        previousResult = None
        for i in range(len(self.layers)-1,-1,-1):
            if(first):
                previousResult = np.array(self.layers[i].backward(output_error))
                if(type(self.layers[i-1]) != FullyConnectedLayer):
                    temp = []
                    for x in range(0,len(previousResult),self.layers[i-1].output_size**2):#for each depth
                        temp.append([])
                        for j in range(0,len(previousResult),self.layers[i-1].output_size):
                            temp[-1].append(previousResult[j:j+self.layers[i-1].output_size])
                    previousResult = np.array(temp)
                first = False
            else:
                previousResult = np.array(self.layers[i].backward(previousResult))