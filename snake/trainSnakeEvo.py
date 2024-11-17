import sys
sys.path.append("../")
import classe
import trainSnakeEvoTools

gameSize = 5

train = trainSnakeEvoTools.trainSnakeEvo(gameSize,20,[75],classe.sigmoid)
train.train()