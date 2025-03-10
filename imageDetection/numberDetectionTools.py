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

KERNELS_SIZE = [3,3,3]

KERNELS_NUMBER = [4,8,16]

LEARNING_RATE = 0.1

BATCH_SIZE = 64

def getNetwork() -> CNN:
    return CNN([
        ConvolutionalLayer(28,28,KERNELS_SIZE[0],KERNELS_NUMBER[0],learning_rate=LEARNING_RATE,activation=Relu,batch_size=BATCH_SIZE),
        BatchNormalization(learning_rate=LEARNING_RATE),
        PoolingLayer(28,14,depth=KERNELS_NUMBER[0],batch_size=BATCH_SIZE),
        ConvolutionalLayer(14,14,KERNELS_SIZE[1],KERNELS_NUMBER[1],depth=KERNELS_NUMBER[0],learning_rate=LEARNING_RATE,activation=Relu,batch_size=BATCH_SIZE),
        BatchNormalization(learning_rate=LEARNING_RATE),
        PoolingLayer(14,7,depth=KERNELS_NUMBER[1],batch_size=BATCH_SIZE),
        ConvolutionalLayer(7,5,KERNELS_SIZE[2],KERNELS_NUMBER[2],depth=KERNELS_NUMBER[1],learning_rate=LEARNING_RATE,activation=Relu,batch_size=BATCH_SIZE),
        BatchNormalization(learning_rate=LEARNING_RATE),
        FlateningLayer(5,KERNELS_NUMBER[-1],batch_size=BATCH_SIZE),
        FullyConnectedLayer(5**2*KERNELS_NUMBER[-1],128,Tanh,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
        Dropout(0.2),
        FullyConnectedLayer(128,10,Sigmoid,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
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