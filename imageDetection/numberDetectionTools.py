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

x_train = x_train.astype('float32')
x_train /= 255

KERNELS_SIZE = [5]

KERNELS_NUMBER = [8]

LEARNING_RATE = 0.002

def getNetwork():
    return CNN([
        ConvolutionalLayer(28,24,KERNELS_SIZE[0],KERNELS_NUMBER[0],learning_rate=LEARNING_RATE,activation=Relu),
        PoolingLayer(24,12,depth=KERNELS_NUMBER[0]),
        FlateningLayer(12,KERNELS_NUMBER[-1]),
        FullyConnectedLayer(12*12*KERNELS_NUMBER[-1],512,Tanh,learningRate=LEARNING_RATE),
        FullyConnectedLayer(512,10,Sigmoid,learningRate=LEARNING_RATE),
    ])

def getTrainedNetwork() -> NN:
    with open(os.path.dirname(os.path.realpath(__file__))+"/numberDetection.pkl", "rb") as file:
        return pickle.load(file)

def getTestData() -> dict[np.array]:
    choice = random.randint(0,len(x_test)-1)
    return dict({"data":x_test[choice].tolist(),"rightAnswer":y_test[choice].tolist()})

def getNetworkAnswer(input : np.array) -> list[float]:
    return getTrainedNetwork().forward(input)