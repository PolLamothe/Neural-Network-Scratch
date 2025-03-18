import argparse
import random
import sys
sys.path.append("../")
from classe import *
import pickle
import numberDetectionTools
import numpy as np

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

if(useTrainedModel):
    network = numberDetectionTools.getTrainedNetwork()
else:
    network = numberDetectionTools.getNetwork()

print("importation of the training dataset is over")

def checkAnswer(right, response):
    greater = None
    for i in range(10):
        if(greater == None):greater = (i,response[i])
        else:
            if(response[i] > greater[1]):greater = (i,response[i])
    if(greater[0] == right):return True
    else:return False

previousAnswerRatio = 0
previousAnswerCoeff = 0

previousAnswerNumber = len(numberDetectionTools.x_test)

if(not useTrainedModel or args.save):
    trainingState = True

    count = 0
    rightCount = dict({})
    for i in range(10):
        rightCount[i] = 0
    while(previousAnswerRatio < 0.95 or previousAnswerCoeff < previousAnswerNumber):
        count += 1
        currentIndexs = [random.randint(0,len(numberDetectionTools.x_train)-1) for i in range(network.batch_size)]
        rights = [numberDetectionTools.y_train[currentIndex] for currentIndex in currentIndexs]
        lastLayerResult = network.forward(np.array([np.append(numberDetectionTools.x_train[currentIndex],[]) for currentIndex in currentIndexs]))

        output_errors = []
        for index,answer in enumerate(lastLayerResult):
            if(checkAnswer(rights[index],answer)):
                previousAnswerRatio = max((previousAnswerRatio*previousAnswerCoeff+1)/(previousAnswerCoeff+1),0)
                previousAnswerCoeff = min(previousAnswerNumber,previousAnswerCoeff+1)
            else:
                previousAnswerRatio = max((previousAnswerRatio*previousAnswerCoeff-1)/(previousAnswerCoeff+1),0)
                previousAnswerCoeff = min(previousAnswerNumber,previousAnswerCoeff+1)
            err = []
            for i in range(10):
                if(i == rights[index]):
                    err.append(1-answer[i])
                else:
                    err.append(-answer[i])
            output_errors.append(copy.deepcopy(err))
        network.backward(np.array(output_errors))
        print("number of iterations : "+str(count*network.batch_size)," success rate : ",round(previousAnswerRatio,2),"   ", end='\r')
        sys.stdout.flush()
    print("\ntraining over")
    trainingState = False


    if(args.save):
        print("saving your model")
        file_name = 'numberDetection.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(network, file)
successCout = 0
testCount = 0
for i in range(len(numberDetectionTools.x_test)):
    result = network.forward([np.append(numberDetectionTools.x_test[i],[])])[0]
    right = numberDetectionTools.y_test[i]
    if(checkAnswer(right,result)):
        successCout += 1
    testCount += 1

print("The success rate of the model on the whole testing dataset is : ",round(successCout/testCount,2))