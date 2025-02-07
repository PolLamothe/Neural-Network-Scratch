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

x_test = x_test.astype('float32')
x_test /= 255

KERNELS_SIZE = [5]

KERNELS_NUMBER = [4]

LEARNING_RATE = 0.002

BATCH_SIZE = 1

def getNetwork() -> CNN:
    return CNN([
        ConvolutionalLayer(28,24,KERNELS_SIZE[0],KERNELS_NUMBER[0],learning_rate=LEARNING_RATE,activation=Relu,batch_size=BATCH_SIZE),
        PoolingLayer(24,12,depth=KERNELS_NUMBER[0],batch_size=BATCH_SIZE),
        FlateningLayer(12,KERNELS_NUMBER[-1],batch_size=BATCH_SIZE),
        FullyConnectedLayer(12*12*KERNELS_NUMBER[-1],512,Tanh,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
        FullyConnectedLayer(512,10,Sigmoid,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
    ],batch_size=BATCH_SIZE)

def getTrainedNetwork() -> CNN:
    with open(os.path.dirname(os.path.realpath(__file__))+"/numberDetection.pkl", "rb") as file:
        return pickle.load(file)

def getTestData() -> dict[np.array]:
    choice = random.randint(0,len(x_test)-1)
    return dict({"data":x_test[choice].tolist(),"rightAnswer":y_test[choice].tolist()})

def getNetworkAnswer(input : np.array) -> dict[list[float]]:
    network = getTrainedNetwork()
    return {
        "answer":network.forward(np.array([input]))[0],
        "convolution":network.layers[0].forward(np.array([input]))[0]
        }