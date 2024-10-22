from keras.datasets import mnist
import pickle
import sys
sys.path.append("../")
from classe import *

#load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:1000]
y_train = y_train[0:1000]

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

def getNetwork():
    return Networks([28*28,50,50,10],sigmoid,0.1)

def getTrainedNetwork():
    with open("numberDetection.pkl", "rb") as file:
        return pickle.load(file)