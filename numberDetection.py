from classe import *
import random
import sys
import threading
from keras.datasets import mnist
import pickle

useTrainedModel = True

if(useTrainedModel):
    with open("numberDetection.pkl", "rb") as file:
        network = pickle.load(file)
else:
    network = Networks([28*28,50,10],sigmoid,0.05)

#load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:1000]
y_train = y_train[0:1000]

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
print("importation over")

def stateChecker():
    while(True):
        input()
        index = random.randint(0,len(x_train)-1)
        data = network.forward(np.append(x_train[index],[]))
        liste = data.tolist()
        for i in range(len(liste)):
            liste[i] = round(liste[i],2)
        if(trainingState):
            print("right answer : "+str(y_train[index])+"\nresult : "+str(liste)+"\nprevious right answer : "+str(rightCount))
        else:
            print("right answer : "+str(y_train[index])+"\nresult : "+str(liste))

def getAccuracy(right,response):
    expected = []
    for i in range(10):
        if(i == right):
            expected.append(1)
        else:
            expected.append(0)
    return np.mean(np.power(expected-response, 2))

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
        currentIndex = random.randint(0,len(x_train)-1)
        right = y_train[currentIndex]
        lastLayerResult = network.forward(np.append(x_train[currentIndex],[]))
        if(checkAnswer(right,lastLayerResult)):
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
        print("number of iterations : "+str(count)+" accuracy : "+str(getAccuracy(right,lastLayerResult)), end='\r')
        sys.stdout.flush()
    print("\ntraining over")
    trainingState = False
    file_name = 'numberDetection.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(network, file)
else:
    trainingState = False