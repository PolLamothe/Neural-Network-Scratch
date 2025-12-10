import sys
import time
sys.path.append("../")
import classe
import trainSnakeEvoTools
import argparse
import pickle
import os
import json
import random
import copy

parser = argparse.ArgumentParser()
parser.add_argument("-c", dest="help", action='store_true')
parser.add_argument("-s", dest="save", action='store_true')
parser.add_argument("-a",dest="aim")
parser.add_argument("-b", dest="benchmark")

args = parser.parse_args()
if(args.help):
    print("-c : Get all command")
    print("-s : Save the model once it has finish training")
    print("-a LENGTH : The average length you want the IA to reach")
    print("-b ID : Benchmarking the agent stored in the file corresponding to the ID provided")
    exit(0)

allModel = []

for (dirpath, dirnames, filenames) in os.walk("./model"):#getting all of the files in model folder
    allModel.extend(filenames)

for i in range(len(allModel)):
    if(allModel[i].split(".")[1] == "pkl"): #keeping only the model files
        allModel[i] = allModel[i].split("_")[1]

with open("./model/trainedData.json","r") as file:
    data : dict= json.load(file)

dataCopy = copy.deepcopy(data)

for Id in dataCopy.keys():#verifying that every data in trainedData.json is linked to an exisisting file
    if(Id not in allModel):
        del data[Id]

with open("./model/trainedData.json","w") as file:
    json.dump(data,file,indent=2)

GAMESIZE = 5

if(args.benchmark != None):
    with open("./model/snake_"+args.benchmark+"_.pkl", "rb") as file:
        network = pickle.load(file)
        trainSnakeEvoTools.trainSnakeEvo.benchmarkModel(network,data[args.benchmark]["gameSize"])
        exit(0)

try:
    if(float(args.aim) > GAMESIZE**2 or float(args.aim) < 4):
        print("the length you want to reach is incorrect")
except TypeError:
    raise Exception("You forgot the parameter -a (press -c to see all comands)")

LEARNING_RATE = 0.01

network = classe.CNN([
    classe.FullyConnectedLayer(GAMESIZE**2*4+9,80,classe.Tanh,LEARNING_RATE,1),
    classe.FullyConnectedLayer(80,80,classe.Tanh,LEARNING_RATE,1),
    classe.FullyConnectedLayer(80,4,classe.Sigmoid,LEARNING_RATE,1)
])

MEAN_SIZE = trainSnakeEvoTools.MEAN_SIZE

ERROR_REVIEW_SIZE = trainSnakeEvoTools.ERROR_REVIEW_SIZE

WINNED_GAME_REVIEW_SIZE = trainSnakeEvoTools.WINNED_GAME_REVIEW_SIZE

PACKED_BODY_COEFF = trainSnakeEvoTools.PACKED_BODY_COEFF

WINNED_GAME_SIZE = trainSnakeEvoTools.WINNED_GAME_SIZE

try:
    STARTINGTIME = time.time()
    snakeTrain = trainSnakeEvoTools.trainSnakeEvo(GAMESIZE,float(args.aim),network)
except TypeError:
    raise Exception("You forgot the parameter -a (press -c to see all comands)")

network = snakeTrain.train()

benchmarkResult = trainSnakeEvoTools.trainSnakeEvo.benchmarkModel(network,GAMESIZE)

if(args.save):
    print("saving your model")
    ID = int(time.time())
    file_name = './model/snake_'+str(ID)+"_.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(network, file)
    with open("./model/trainedData.json","r") as file:
        data = json.load(file)
    data[ID] = dict({
        "gameSize":GAMESIZE,
        "aim":args.aim,
        "trainingTime":(time.time()-STARTINGTIME)/60,
        "LEARNING_RATE" : LEARNING_RATE,
        "MEAN_SIZE" : MEAN_SIZE,
        "ERROR_REVIEW_SIZE" : ERROR_REVIEW_SIZE,
        "WINNED_GAME_REVIEW_SIZE" : WINNED_GAME_REVIEW_SIZE,
        "PACKED_BODY_COEFF" : PACKED_BODY_COEFF,
        "WINNED_GAME_SIZE" : WINNED_GAME_SIZE,
        "HIDDEN_LAYER" : len(network.layers)-1,
        "BENCHMARK" : benchmarkResult
    })
    with open("./model/trainedData.json","w") as file:
        json.dump(data,file,indent=2)


os._exit(0)