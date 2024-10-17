import numpy as np
import math
from time import sleep
import random
import sys
from classe import *
import threading

data = []
for i in range(2):
    for x in range(2):
        for j in range(2):
            if(i == 0 and x == 0 and j == 0):
                continue
            data.append(np.array([i,x,j]))

def checkResponse(array,response):
    expected = getRightAnswer(array)
    if(expected == 1):
        if(response >= 0.95):
            return True
    else:
        if(response <= 0.05):
            return True
    return False

def getRightAnswer(array):
    return sum(array) == 2

def stateChecker():
    while(True):
        input()
        choice = random.choice(data)
        firstLayerResult = firstLayer.forward(currentData)
        lastLayerResult = lastLayer.forward(firstLayerResult)[0]
        print(choice,lastLayerResult)

thread = threading.Thread(target=stateChecker)
thread.start()

learningRate = 0.1
activation = tanh
layers = [3,10,1]

firstLayer = Layer(layers[0],layers[1],activation,learningRate)
lastLayer = Layer(layers[1],layers[2],activation,learningRate)

#for neurones in (firstLayer.nerone+lastLayer.nerone):print(neurones.W,neurones.bias)

count = 0
while(True):
    count += 1
    currentData = random.choice(data)
    firstLayerResult = firstLayer.forward(currentData)
    lastLayerResult = lastLayer.forward(firstLayerResult)[0]
    if(checkResponse(currentData,lastLayerResult)):
        state = True
        for array in data:
            firstLayerResult = firstLayer.forward(array)
            resultTemp = lastLayer.forward(firstLayerResult)[0]
            if(not checkResponse(array,resultTemp)):
                state = False
        if(state):
            break
    err = np.array([(getRightAnswer(currentData)-lastLayerResult)])
    err = lastLayer.backward(err)
    firstLayer.backward(err)
    print("number of iterations : "+str(count), end='\r')
    sys.stdout.flush()
print("\nTraining is over !")

for array in data:
    firstLayerResult = firstLayer.forward(array)
    lastLayerResult = lastLayer.forward(firstLayerResult)[0]
    print(array,round(lastLayerResult,2))