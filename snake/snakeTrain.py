import sys
sys.path.append("../")
import snakeTrainTools
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
seeAllMap = args.fullMap

snakeTrain = snakeTrainTools.snakeTrainTools(gameSize,seeAllMap,int(args.aim),[75])
snakeTrain.train()

if(args.save):
    print("saving your model")
    file_name = './model/snake_'+str(gameSize)
    if(seeAllMap):file_name+="_FullMap"
    else:file_name+="_NearHead"
    file_name+=".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(snakeTrain.network, file)

os._exit(0)