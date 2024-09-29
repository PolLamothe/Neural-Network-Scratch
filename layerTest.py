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
    if(abs(response-expected) <= 0.01):
        return True
    return False

def getRightAnswer(array):
    if((array[0] == 1 or (array[1]+array[2] == 2)) and(sum(array) != 3)):
        return 1
    return 0

def stateChecker():
    while(True):
        input()
        choice = random.choice(data)
        print(choice,network.forward(choice))

thread = threading.Thread(target=stateChecker)
thread.start()

network = Networks([3,10,1],0.3)

count = 0
while(True):
    count += 1
    currentData = random.choice(data)
    lastLayerResult = network.forward(currentData)[0]
    if(checkResponse(currentData,lastLayerResult)):
        state = True
        for array in data:
            resultTemp = network.forward(array)
            if(not checkResponse(array,resultTemp)):
                state = False
        if(state):
            break
    err = np.array([(getRightAnswer(currentData)-lastLayerResult)])
    network.backward(err)
    print("number of iterations : "+str(count), end='\r')
    sys.stdout.flush()
thread = None
print("\nTraining is over !")

for array in data:
    lastLayerResult = network.forward(array)
    print(array,round(lastLayerResult[0],2))