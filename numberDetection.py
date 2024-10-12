import argparse
from classe import *
import random
import sys
import threading
import pickle
import numberDetectionTools

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


print("importation over")

def stateChecker():
    while(True):
        input()
        index = random.randint(0,len(numberDetectionTools.x_train)-1)
        data = network.forward(np.append(numberDetectionTools.x_train[index],[]))
        liste = data.tolist()
        for i in range(len(liste)):
            liste[i] = round(liste[i],2)
        if(trainingState):
            print("right answer : "+str(numberDetectionTools.y_train[index])+"\nresult : "+str(liste)+"\nprevious right answer : "+str(rightCount))
        else:
            print("right answer : "+str(numberDetectionTools.y_train[index])+"\nresult : "+str(liste))

def checkAnswer(right, response):
    for i in range(10):
        if(i ==right):
            if(response[i] < 0.8):return False
        else:
            if(response[i] > 0.2):return False
    return True

thread = threading.Thread(target=stateChecker)
thread.start()

if(not useTrainedModel):
    trainingState = True

    count = 0
    rightCount = dict({})
    for i in range(10):
        rightCount[i] = 0
    while(True):
        count += 1
        currentIndex = random.randint(0,len(numberDetectionTools.x_train)-1)
        right = numberDetectionTools.y_train[currentIndex]
        lastLayerResult = network.forward(np.append(numberDetectionTools.x_train[currentIndex],[]))
        if(checkAnswer(right,lastLayerResult)):
            if(rightCount[right] <= 50):
                rightCount[right] += 1
            state = True
            for i in range(10):
                if(rightCount[i] < 10):
                    state = False
                    break
            if(state):
                break
        else:
            if(rightCount[right] > 0):rightCount[right] -= 1
        err = []
        for i in range(10):
            if(i != right):
                err.append(0-lastLayerResult[i])
            else:
                err.append(1-lastLayerResult[i])
        network.backward(err)
        print("number of iterations : "+str(count), end='\r')
        sys.stdout.flush()
    print("\ntraining over")
    trainingState = False
    if(args.save):
        print("saving your model")
        file_name = 'numberDetection.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(network, file)
else:
    trainingState = False