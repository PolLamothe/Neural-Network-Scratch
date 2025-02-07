from tensorflow.keras.datasets import mnist
import random
import pickle
import sys
sys.path.append("../")  
from classe import *
import os 
import numpy as np

#load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:7000]
y_train = y_train[0:7000]

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

def getNetwork() -> FNN:
    return FNN([28*28,100,10],neuroneActivation=[Tanh,Sigmoid],learningRate=0.1,batch_size=16)

def getTrainedNetwork() -> FNN:
    with open(os.path.dirname(os.path.realpath(__file__))+"/numberDetection.pkl", "rb") as file:
        return pickle.load(file)

def getTestData() -> dict[np.array]:
    choice = random.randint(0,len(x_test)-1)
    return dict({"data":x_test[choice].tolist(),"rightAnswer":y_test[choice].tolist()})

def getNetworkAnswer(input : np.array) -> list[float]:
    return getTrainedNetwork().forward(np.array([input]))[0]