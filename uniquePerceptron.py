import numpy as np
import threading
from time import sleep
import random
import sys
from classe import *

data = []
for i in range(2):
    for x in range(2):
        for j in range(2):
            if(i == 0 and x == 0 and j == 0):
                continue
            data.append(np.array([i,x,j]))

def checkResponse(array,response):
    expected = getRightAnswer(array)
    if(abs(response-expected) <= 0.01):
        return True
    return False

def getRightAnswer(array):
    return array[0]

def stateChecker():
    while(True):
        input()
        choice = random.choice(data)
        print(choice,perceptron.forward(choice),perceptron.W)

thread = threading.Thread(target=stateChecker)
thread.start()

perceptron = Layer(3,1,relu,0.1)

#training
count = 0
while True:
    count += 1
    currentData = random.choice(data)
    result = perceptron.forward(np.array(currentData))
    if(checkResponse(currentData,result)):
        state = True
        for array in data:
            resultTemp = perceptron.forward(array)
            if(not checkResponse(array,resultTemp)):
                state = False
        if(state):
            break
    err = (getRightAnswer(currentData)-result)
    perceptron.backward(err)
    print("number of iterations : "+str(count), end='\r')
    sys.stdout.flush()
print("\ntraining over")

#test
for array in data:
    result = perceptron.forward(array)
    print(array,np.round(result,2))