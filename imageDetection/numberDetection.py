import argparse
import random
import sys
sys.path.append("../")
from classe import *
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

print("importation of the training dataset is over")

def checkAnswer(right, response):
    greater = None
    for i in range(10):
        if(greater == None):greater = (i,response[i])
        else:
            if(response[i] == greater[1]):return False
            elif(response[i] > greater[1]):greater = (i,response[i])
    if(greater[0] == right):return True
    else:return False

def changeFormat(data : np.ndarray):
    result : list[list]= []
    for i in range(len(data)):
        if(i%28 == 0):
            result.append([])
        result[-1].append(data[i])
    return np.array(result)

previousAnswerRatio = 0
previousAnswerCoeff = 0

previousAnswerNumber = 1000

if(not useTrainedModel or args.save):
    trainingState = True

    count = 0
    while(previousAnswerRatio < 0.9 or previousAnswerCoeff < previousAnswerNumber):
        count += 1
        currentIndexs = [random.randint(0,len(numberDetectionTools.x_train)-1) for i in range(network.batch_size)]
        rights = [numberDetectionTools.y_train[currentIndex] for currentIndex in currentIndexs]
        lastLayerResult = network.forward(np.array([[numberDetectionTools.x_train[currentIndex]] for currentIndex in currentIndexs]))

        output_errors = []
        mse_sum = 0
        for index,answer in enumerate(lastLayerResult):
            if(checkAnswer(rights[index],answer)):
                previousAnswerRatio = max((previousAnswerRatio*previousAnswerCoeff+1)/(previousAnswerCoeff+1),0)
                previousAnswerCoeff = min(previousAnswerNumber,previousAnswerCoeff+1)
            else:
                previousAnswerRatio = max((previousAnswerRatio*previousAnswerCoeff)/(previousAnswerCoeff+1),0)
                previousAnswerCoeff = min(previousAnswerNumber,previousAnswerCoeff+1)
            err = []
            for i in range(10):
                if(i == rights[index]):
                    err.append(1-answer[i])
                    mse_sum += ((1-answer[i])**2)/10/numberDetectionTools.BATCH_SIZE
                else:
                    err.append(-answer[i])
                    mse_sum += ((-answer[i])**2)/10/numberDetectionTools.BATCH_SIZE
            output_errors.append(copy.deepcopy(err))
        network.backward(np.array(output_errors))
        #print(round(np.sum(network.layers[0].error)/numberDetectionTools.BATCH_SIZE,3))
        print("nombre d'iterations : "+str(count*numberDetectionTools.BATCH_SIZE)," mse : ",round(mse_sum,2)," success rate : ",round(previousAnswerRatio,2),"    ")
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
    result = network.forward(np.array([[numberDetectionTools.x_test[i]]]))[0]
    right = numberDetectionTools.y_test[i]
    if(checkAnswer(right,result)):
        successCout += 1
    testCount += 1

print("The success rate of the model on the whole testing dataset is : ",round(successCout/testCount,2))