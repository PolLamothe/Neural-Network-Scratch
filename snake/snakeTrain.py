import sys
sys.path.append("../")
import snakeTrainTools
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("-c", dest="help", action='store_true')
parser.add_argument("-s", dest="save", action='store_true')
args = parser.parse_args()
if(args.help):
    print("-c : Get all command")
    print("-s : Save the model once it has finish training")
    exit(0)

gameSize = 5
seeAllMap = True

snakeTrain = snakeTrainTools.snakeTrainTools(gameSize,seeAllMap)
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