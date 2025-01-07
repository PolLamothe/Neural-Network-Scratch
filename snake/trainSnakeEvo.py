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
parser.add_argument("-f",dest="fullMap", action="store_true")
parser.add_argument("-a",dest="aim")

args = parser.parse_args()
if(args.help):
    print("-c : Get all command")
    print("-s : Save the model once it has finish training")
    print("-f : Give all of the grid to the IA")
    print("-a : The average length you want the IA to reach")
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

gameSize = 5

HIDDENLAYERS = [150]

try:
    if(int(args.aim) > gameSize**2 or int(args.aim) < 4):
        print("the length you want to reach is incorrect")
except TypeError:
    raise Exception("You forgot the parameter -a (press -c to see all comands)")

try:
    STARTINGTIME = time.time()
    snakeTrain = trainSnakeEvoTools.trainSnakeEvo(gameSize,args.aim,HIDDENLAYERS,neuroneActivation=[classe.tanh,classe.sigmoid])
except TypeError:
    raise Exception("You forgot the parameter -a (press -c to see all comands)")

SELECTIONSIZE = trainSnakeEvoTools.SELECTIONSIZE
MULTIPLIER = trainSnakeEvoTools.MULTIPLIER
ITERATION = trainSnakeEvoTools.ITERATION

network = snakeTrain.train()

if(args.save):
    print("saving your model")
    ID = int(time.time())
    file_name = './model/snake_'+str(ID)+"_.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(network, file)
    with open("./model/trainedData.json","r") as file:
        data = json.load(file)
    data[ID] = dict({
        "gameSize":gameSize,
        "aim":args.aim,
        "hiddenLayers":HIDDENLAYERS,
        "activationFunction":str(snakeTrain.activationFunction),
        "neuroneActivation":str(snakeTrain.neuroneActivation),
        "trainingTime":(time.time()-STARTINGTIME)/60,
        "selectionSize" : SELECTIONSIZE,
        "multiplier" : MULTIPLIER,
        "iteration" : ITERATION,
    })
    with open("./model/trainedData.json","w") as file:
        json.dump(data,file,indent=2)

os._exit(0)