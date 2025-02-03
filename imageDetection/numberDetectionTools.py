from keras.datasets import mnist
import random
import pickle
import sys
sys.path.append("../")  
from classe import *
import os 

#load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:7000]
y_train = y_train[0:7000]

KERNELS_SIZE = [3]

KERNELS_NUMBER = [16]

def getNetwork():
    return CNN([
        ConvolutionalLayer(28,28,KERNELS_SIZE[0],KERNELS_NUMBER[0],learning_rate=0.1,activation=Relu),PoolingLayer(28,14,depth=KERNELS_NUMBER[0]),
        FullyConnectedLayer(14*14*KERNELS_NUMBER[0],10,Sigmoid,learningRate=0.1)])

def getTrainedNetwork() -> NN:
    with open(os.path.dirname(os.path.realpath(__file__))+"/numberDetection.pkl", "rb") as file:
        return pickle.load(file)

def getTestData() -> dict[np.array]:
    choice = random.randint(0,len(x_test)-1)
    return dict({"data":x_test[choice].tolist(),"rightAnswer":y_test[choice].tolist()})

def getNetworkAnswer(input : np.array) -> list[float]:
    return getTrainedNetwork().forward(input)