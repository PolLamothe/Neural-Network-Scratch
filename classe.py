import math
import numpy as np
import scipy.signal
import copy

class ActivationFunction():
    def function(X):
        return X
    
    def derivative(X):
        return 1
    
class Sigmoid(ActivationFunction):
    def function(X):
        X = np.clip(X,-600,600)
        return 1/(1+np.exp(-X))
    
    def derivative(X):
        X = np.clip(X,-600,600)
        sigm = Sigmoid.function(X)
        return sigm*(1-sigm)

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
    
class Relu(ActivationFunction):
    def function(X):
        return np.maximum(X,0)
    
    def derivative(X):
        return np.minimum(X,1)
    
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
    def __init__(self, input_size: int, output_size: int, activation, learningRate: float = 1, batch_size: int = 1, parents: list = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.learningRate = learningRate
        self.batch_size = batch_size

        if parents is None:
            self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / (input_size + output_size))
            self.B = np.random.uniform(low=-1, high=1, size=(output_size,))
        else:
            self.W = (parents[0].W + parents[1].W) / 2
            self.B = (parents[0].B + parents[1].B) / 2

    def forward(self, this_input: np.ndarray) -> np.ndarray:
        self.X = this_input
        self.Y = self.activation.function(np.dot(this_input, self.W.T) + self.B)
        return self.Y

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        output_error *= self.activation.derivative(self.Y)

        grad_W = np.zeros_like(self.W)
        for i in range(self.X.shape[0]):
            grad_W += (output_error[i][:, np.newaxis] * self.X[i]) / self.X.shape[0]
        grad_B = np.zeros_like(self.B)
        for i in range(self.X.shape[0]):
            grad_B += output_error[i] / self.X.shape[0]

        self.W += self.learningRate * grad_W
        self.B += self.learningRate * grad_B

        return np.dot(output_error, self.W)

class ConvolutionalLayer(Layer):
    def __init__(self,input_size : int,output_size : int,kernel_size : int,kernel_number : int,activation : ActivationFunction,depth : int = 1,learning_rate : float = 1,batch_size :int = 1):
        self.input_size = input_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.kernel_number = kernel_number
        self.selection_size = self.input_size - self.kernel_size +1
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_size = output_size
        self.batch_size = batch_size

        self.P = self.output_size - self.selection_size
        if(self.P%2 != 0):raise Exception("This padding isn't supported")

        self.K = []
        self.B = np.random.randn(self.kernel_number) * np.sqrt(2 / self.kernel_number)

        for j in range(self.kernel_number):
            self.K.append([])
            #self.B.append(np.random.randn(self.selection_size, self.selection_size) * np.sqrt(2. / (self.selection_size + self.selection_size)))
            for i in range(self.depth):    
                #self.K[-1].append(np.random.randn(kernel_size, kernel_size) * np.sqrt(2. / (kernel_size + kernel_size)))
                #Initialisation He
                self.K[-1].append(np.random.normal(0, np.sqrt(2.0 / (self.kernel_size**2 * self.depth)), (self.kernel_size,self.kernel_size)))
        self.K = np.array(self.K)

    def forward(self,this_input : np.ndarray) -> np.ndarray:
        result = []
        self.X = this_input

        this_output = []

        for x in range(self.X.shape[0]):
            this_output.append([])
            for i in range(self.kernel_number):
                somme = np.zeros((self.selection_size,self.selection_size))
                for j in range(self.depth):
                    somme += scipy.signal.correlate2d(
                        self.X[x][j],
                        self.K[i][j],mode="valid")
                somme += self.B[i]
                this_output[-1].append(copy.deepcopy(somme))
        
        if(self.P != 0):
            for i in range(self.X.shape[0]):
                for j in range(self.kernel_number):
                    this_output[i][j] = np.pad(this_output[i][j], pad_width=int(self.P/2), mode='constant', constant_values=0)

        this_output = self.activation.function(this_output)

        self.Y = np.array(this_output)
        return self.Y
    
    def backward(self,error : np.ndarray) -> np.ndarray:
        self.error = error
        inputWithoutPad = []
        errorWithoutPad = []
        if(self.P != 0):
            for i in range(self.Y.shape[0]):
                inputWithoutPad.append([])
                errorWithoutPad.append([])
                for j in range(self.kernel_number):
                    inputWithoutPad[-1].append(self.Y[i][j][int(self.P/2) : -int(self.P/2), int(self.P/2) : -int(self.P/2)])
                    errorWithoutPad[-1].append(self.error[i][j][int(self.P/2) : -int(self.P/2), int(self.P/2) : -int(self.P/2)])
            inputWithoutPad = np.array(inputWithoutPad)
            errorWithoutPad = np.array(errorWithoutPad)
        else:
            inputWithoutPad = self.Y
            errorWithoutPad = error
        errorWithoutPad *= self.activation.derivative(inputWithoutPad)

        kernels_gradiant = np.zeros_like(self.K)

        for x in range(self.X.shape[0]):
            for j in range(self.kernel_number):
                for i in range(self.depth):
                    kernels_gradiant[j][i] += scipy.signal.correlate2d(
                        self.X[x][i],
                        errorWithoutPad[x][j]
                    ,mode="valid")

                self.B[j] += np.sum(errorWithoutPad[x][j]) * self.learning_rate / self.X.shape[0]
        
        input_error : list[np.ndarray] = []
        for x in range(self.X.shape[0]):
            input_error.append([])
            for i in range(self.depth):
                temp = np.zeros((self.input_size,self.input_size))
                for j in range(self.kernel_number):
                    temp += scipy.signal.convolve2d(
                        errorWithoutPad[x][j],
                        self.K[j][i],mode="full")
                input_error[-1].append(copy.deepcopy(temp))
        input_error = np.array(input_error)

        self.K += kernels_gradiant * (self.learning_rate / self.X.shape[0])

        return input_error

class PoolingLayer(Layer):
    def __init__(self,input_size : int,output_size : int, max_pooling : bool = True,depth : int = 1,batch_size : int = 1):
        self.input_size = input_size
        self.output_size = output_size
        self.max_pooling = max_pooling
        self.depth = depth
        self.batch_size = batch_size
        if(self.input_size%self.output_size != 0):
            raise Exception("The size of the layer is not valid")
        self.selection_size = self.input_size//self.output_size
    
    def forward(self,this_input : np.ndarray) -> np.ndarray:
        self.X = copy.deepcopy(this_input)
        self.Y = []
        for batch in range(self.X.shape[0]):
            self.Y.append([])
            for i in range(self.depth):
                temp = np.zeros((self.output_size,self.output_size))
                for j in range(self.input_size//self.selection_size):
                    for x in range(self.input_size//self.selection_size):
                        region = this_input[batch][i][
                            j*self.selection_size:(j+1)*self.selection_size,
                            x*self.selection_size:(x+1)*self.selection_size,
                            ]
                        if(self.max_pooling):
                            temp[j,x] = np.max(region)
                        else:
                            temp[j,x] = np.mean(region)
                self.Y[-1].append(copy.deepcopy(temp))
        self.Y = np.array(self.Y)

        return self.Y
    
    def backward(self,error : np.ndarray) -> np.ndarray:
        result = []
        for batch in range(error.shape[0]):
            result.append([])
            for i in range(self.depth):
                result[-1].append(np.zeros(self.X[batch][i].shape))
                for j in range(0,self.input_size,self.selection_size):
                    for x in range(0,self.input_size,self.selection_size):
                        region = self.X[batch][i][j:j+self.selection_size,x:x+self.selection_size]
                        if(self.max_pooling):
                            max_index = np.unravel_index(region.argmax(), region.shape)
                            result[-1][-1][j+max_index[0]][x+max_index[1]] = error[batch][i][j//self.selection_size,x//self.selection_size]
                        else:
                            result[-1][-1][j:j+self.selection_size,x:x+self.selection_size] += error[batch][i][j//self.selection_size,x//self.selection_size] / (self.selection_size**2)
        return np.array(result)
    
class FlateningLayer(Layer):
    def __init__(self,input_size : int,input_depth : int,batch_size : int= 1):
        self.input_size = input_size
        self.input_depth = input_depth
        self.batch_size = batch_size

    def forward(self,this_input : np.ndarray) -> np.ndarray:
        result = []
        for j in range(this_input.shape[0]):
            result.append([])
            for i in range(self.input_depth):
                result[-1].extend(np.append(this_input[j][i],[]))
        return np.array(result)

    def backward(self,this_error : np.ndarray) -> np.ndarray:
        result = []
        for x in range(this_error.shape[0]):
            result.append([])
            for i in range(self.input_depth):
                temp = []
                for j in range(self.input_size):
                    base = i*(self.input_size**2)
                    temp.append(this_error[x][base+j*self.input_size:base+(j+1)*self.input_size])
                result[-1].append(copy.deepcopy(temp))
        result = np.array(result)
        return result
    
class BatchNormalization(Layer):
    def __init__(self,learning_rate : float):
        self.gamma = 1
        self.beta = 0
        self.epsilon = 1e-15
        self.learning_rate = learning_rate

    def forward(self,X : np.ndarray) -> np.ndarray:
        self.X = X
        
        means = np.mean(X, axis=(0, 2, 3), keepdims=True)
        vars = np.var(X, axis=(0, 2, 3), keepdims=True)
        
        self.x_hat = (X - means) / np.sqrt(vars + self.epsilon)
        
        self.Y = self.gamma * self.x_hat + self.beta

        self.means = means
        self.vars = vars
        
        return self.Y
            
    def backward(self,error : np.ndarray) -> np.ndarray:
        N, C, H, W = error.shape
        m = N * H * W

        dGamma = np.sum(error * self.x_hat, axis=(0, 2, 3), keepdims=True)
        dBeta = np.sum(error, axis=(0, 2, 3), keepdims=True)

        self.gamma += self.learning_rate * dGamma
        self.beta  += self.learning_rate * dBeta

        error_mean = np.mean(error, axis=(0, 2, 3), keepdims=True)
        error_xhat_mean = np.mean(error * self.x_hat, axis=(0, 2, 3), keepdims=True)
        
        dX = (self.gamma / np.sqrt(self.vars + self.epsilon)) * (error - error_mean - self.x_hat * error_xhat_mean)
        
        return dX


class FNN(NN):
    def __init__(self,neuroneNumber : list[int],learningRate : float=1,neuroneActivation : list[ActivationFunction]=None,batch_size : int = 1,parents : list=None) -> None:
        self.layers : list[FullyConnectedLayer] = []
        self.batch_size = batch_size
        activationList = []
        for i in range(len(neuroneActivation)):
            activationList.append(neuroneActivation[i])
        for i in range(1,len(neuroneNumber)):
            try:
                if(parents != None):
                    self.layers.append(FullyConnectedLayer(neuroneNumber[i-1],neuroneNumber[i],activationList[i-1],learningRate,parents=[parents[0].layers[i-1],parents[1].layers[i-1]],batch_size=batch_size))
                else:
                    self.layers.append(FullyConnectedLayer(neuroneNumber[i-1],neuroneNumber[i],activationList[i-1],learningRate,batch_size=batch_size))
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
    def __init__(self,layers : list[Layer],batch_size : int = 1):
        self.layers = layers
        self.batch_size = batch_size

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
        return previousResult