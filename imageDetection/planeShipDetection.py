import copy
import os
import pickle
from PIL import Image
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import classe
import time
import random
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", dest="loadTrainedModel", action='store_true')
parser.add_argument("-c", dest="help", action='store_true')
parser.add_argument("-s", dest="save", action='store_true')
args = parser.parse_args()
useTrainedModel = args.loadTrainedModel
if(args.help):
    print("-c : Get all command")
    print("-l : Use the trained model")
    print("-s : Save the model once it has finish training")
    exit(0)

def getTrainedNetwork() -> classe.CNN:
    with open(os.path.dirname(os.path.realpath(__file__))+"/planeShipDetection.pkl", "rb") as file:
        return pickle.load(file)

PATH_TO_TRAINING = "./assets/plane-and-ship/train/"

TRAINING_SIZE = 1000

if(TRAINING_SIZE > 1000):
    raise Exception("The training size is too high")

trainingDataSet : list[list] = [[],[]]


if(not args.loadTrainedModel):
    print("starting training data serialization")

    for i in range(2):
        if(i == 0):currentObject = "airplanes"
        else:currentObject = "ship"

        dossier = Path(PATH_TO_TRAINING+currentObject)
        fichiers = [f.name for f in dossier.iterdir() if f.is_file()]

        for j in range(0,TRAINING_SIZE):
            image = Image.open(PATH_TO_TRAINING+currentObject+'/'+fichiers[j]).convert("RGB")

            matrix = np.array(image)

            resized_matrix = np.stack([scipy.ndimage.zoom(matrix[:, :, i], (64/matrix.shape[0],64/matrix.shape[1])) for i in range(3)], axis=-1)

            resized_matrix =  np.transpose(resized_matrix, (2, 0, 1))

            trainingDataSet[i].append(resized_matrix/255)

    trainingDataSet = np.array(trainingDataSet)

    print("training data serialized")

KERNELS_SIZE = [3,3,3,3]

KERNELS_NUMBER = [16,32,64,128]

LEARNING_RATE = 0.05

BATCH_SIZE = 64

if(useTrainedModel):
    network = getTrainedNetwork()
else:
    network = classe.CNN([
        classe.ConvolutionalLayer(64,64,KERNELS_SIZE[0],KERNELS_NUMBER[0],depth=3,learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
        classe.BatchNormalization(learning_rate=LEARNING_RATE),
        classe.PoolingLayer(64,32,depth=KERNELS_NUMBER[0],batch_size=BATCH_SIZE),
        classe.ConvolutionalLayer(32,32,KERNELS_SIZE[1],KERNELS_NUMBER[1],depth=KERNELS_NUMBER[0],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
        classe.BatchNormalization(learning_rate=LEARNING_RATE),
        classe.PoolingLayer(32,16,depth=KERNELS_NUMBER[1],batch_size=BATCH_SIZE),
        classe.ConvolutionalLayer(16,16,KERNELS_SIZE[2],KERNELS_NUMBER[2],depth=KERNELS_NUMBER[1],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
        classe.BatchNormalization(learning_rate=LEARNING_RATE),
        classe.PoolingLayer(16,8,depth=KERNELS_NUMBER[2],batch_size=BATCH_SIZE),
        classe.ConvolutionalLayer(8,8,KERNELS_SIZE[3],KERNELS_NUMBER[3],depth=KERNELS_NUMBER[2],learning_rate=LEARNING_RATE,activation=classe.Relu,batch_size=BATCH_SIZE),
        classe.BatchNormalization(learning_rate=LEARNING_RATE),
        classe.PoolingLayer(8,4,depth=KERNELS_NUMBER[3],batch_size=BATCH_SIZE),
        classe.FlateningLayer(4,KERNELS_NUMBER[-1],batch_size=BATCH_SIZE),
        classe.FullyConnectedLayer(4**2*KERNELS_NUMBER[-1],256,classe.Tanh,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
        classe.FullyConnectedLayer(256,2,classe.Sigmoid,learningRate=LEARNING_RATE,batch_size=BATCH_SIZE),
    ])


def checkAnswer(right, response):
    greater = None
    for i in range(2):
        if(greater == None):greater = (i,response[i])
        else:
            if(response[i] == greater[1]):return False
            elif(response[i] > greater[1]):greater = (i,response[i])
    if(greater[0] == right):return True
    else:return False

previousAnswerRatio = 0
previousAnswerCoeff = 0

count = 0

COEFF_SIZE = 300

startTime = time.time()

if(not args.loadTrainedModel):
    while(previousAnswerRatio < 0.85 or previousAnswerCoeff < COEFF_SIZE):
        count += 1
        currentIndexs = [random.randint(0,TRAINING_SIZE-1) for i in range(BATCH_SIZE)]
        rights = [random.randint(0,1) for i in range(BATCH_SIZE)]

        lastLayerResult = network.forward(np.array([trainingDataSet[rights[i]][currentIndexs[i]] for i in range(BATCH_SIZE)]))
        error = []
        mse_sum = 0

        if(lastLayerResult[0][0] == 0):
            exit(1)

        for index,value in enumerate(lastLayerResult):
            right = rights[index]
            if(checkAnswer(right,value)):
                previousAnswerRatio = (previousAnswerRatio*previousAnswerCoeff+1)/(previousAnswerCoeff+1)
            else:
                previousAnswerRatio = (previousAnswerRatio*previousAnswerCoeff)/(previousAnswerCoeff+1)
            previousAnswerCoeff = min(COEFF_SIZE,previousAnswerCoeff+1)
            err = []
            for i in range(2):
                if(i == right):
                    err.append((1-value[i])*abs((1-value[i])))
                    mse_sum += ((1-value[i])**2)/2/BATCH_SIZE
                else:
                    err.append(-value[i]*abs((-value[i])))
                    mse_sum += ((-value[i])**2)/2/BATCH_SIZE
            error.append(copy.deepcopy(err))
        network.backward(np.array(error))
        print("nombre d'iterations : "+str(count*BATCH_SIZE)," mse : ",round(mse_sum,2)," success rate : ",round(previousAnswerRatio,2))
        sys.stdout.flush()
    print("\ntraining over")
    
    if(args.save):
        print("saving your model")
        file_name = 'planeShipDetection.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(network, file)

PATH_TO_TEST = "./assets/plane-and-ship/test/"

testingDataSet : list[list] = [[],[]]

print("starting testing data serialization")

for i in range(2):
    if(i == 0):currentObject = "airplanes"
    else:currentObject = "ships"

    dossier = Path(PATH_TO_TEST+currentObject)
    fichiers = [f.name for f in dossier.iterdir() if f.is_file()]

    for j in range(0,len(fichiers)-1):
        image = Image.open(PATH_TO_TEST+currentObject+'/'+fichiers[j]).convert("RGB")

        matrix = np.array(image)

        resized_matrix = np.stack([scipy.ndimage.zoom(matrix[:, :, i], (64/matrix.shape[0],64/matrix.shape[1])) for i in range(3)], axis=-1)

        resized_matrix =  np.transpose(resized_matrix, (2, 0, 1))

        testingDataSet[i].append(resized_matrix/255)

testingDataSet[0] = testingDataSet[0][:min(len(testingDataSet[0]),len(testingDataSet[1]))]
testingDataSet[1] = testingDataSet[1][:min(len(testingDataSet[0]),len(testingDataSet[1]))]

testingDataSet = np.array(testingDataSet)

print("testing data serialized")

successCout = 0
testCount = 0

for j in range(2):
    for i in range(len(testingDataSet[j])):
        result = network.forward(np.array([testingDataSet[j][i]]))[0]
        if(checkAnswer(j,result)):
            successCout += 1
        testCount += 1

print("The success rate of the model on the whole testing dataset is : ",round(successCout/testCount,2))