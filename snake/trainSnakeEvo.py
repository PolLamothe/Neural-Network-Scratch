import sys
sys.path.append("../")
import classe
import trainSnakeEvoTools
import argparse
import pickle
import os

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

gameSize = 5

snakeTrain = trainSnakeEvoTools.trainSnakeEvo(gameSize,args.aim,[160],classe.sigmoid)
snakeTrain.train()

if(args.save):
    print("saving your model")
    file_name = './model/snake_'+str(gameSize)
    file_name+="_FullMap"
    file_name+=".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(snakeTrain.network, file)

os._exit(0)