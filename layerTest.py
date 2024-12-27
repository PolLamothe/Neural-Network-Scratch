import copy
import json
import numpy as np
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
    if(expected == 1):
        if(response >= 0.95):
            return True
    else:
        if(response <= 0.05):
            return True
    return False

def getRightAnswer(array):
    return sum(array) == 2

network = Networks([3,3,1],learningRate=0.5,neuroneActivation=[sigmoid,sigmoid])

count = 0
while(True):
    count += 1
    currentData = random.choice(data)
    lastLayerResult = network.forward(currentData)[0]

    if(checkResponse(currentData,lastLayerResult)):
        state = True
        for array in data:
            resultTemp = network.forward(np.array(array))[0]
            if(not checkResponse(array,resultTemp)):
                state = False
        if(state):
            break
    err = np.array([(getRightAnswer(currentData)-lastLayerResult)])
    network.backward(err)
    print("number of iterations : "+str(count), end='\r')
    sys.stdout.flush()
print("\nTraining is over !")

dataSaved = []
for array in data:
    lastLayerResult = network.forward(array)[0]
    temp = [network.layers[0].X.tolist()]
    for layer in network.layers:
        temp.append(layer.Y.tolist())
    dataSaved.append(copy.deepcopy(temp))
    print(array,round(lastLayerResult,2))

with open("./web/static/trainedData.json","w") as file:
    json.dump(dataSaved,file,indent=2)