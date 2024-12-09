import os
import snake.snakeGame as snakeGame
import snake.trainSnakeEvoTools as trainSnakeEvoTools
import pickle
import random
import sys
sys.path.append("../")
import classe
import copy
import json

allModel = []
for (dirpath, dirnames, filenames) in os.walk("./model"):
    allModel.extend(filenames)

for model in allModel:
    if(model.split(".")[1] != "pkl"): #keeping only the model files
        allModel.remove(model)
    else:
        model = model.split("_")[2]

ui = snakeGame.UI()

game = None

def handleModelChoice(name : str):
    global dataSinceLastFood
    global ui
    global game
    with open("./model/"+name, "rb") as file:
        network : classe.Networks = pickle.load(file)
    
    with open("./model/trainedData.json","r") as file:
        data : dict= json.load(file)
    gameSize = int(data[name.split("_")[1]]["gameSize"])

    game = snakeGame.Game(gameSize)
    dataSinceLastFood = []

    def updateGrid():
        global dataSinceLastFood
        global game
        result = network.forward(trainSnakeEvoTools.trainSnakeEvo.generateInput(game.getGrid(),True,game.snake))
        answerIndex = random.choice(trainSnakeEvoTools.trainSnakeEvo.getAllMaxIndex(trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(game.size,game.snake,result,copy.deepcopy(dataSinceLastFood))))
        if(answerIndex == 0):
            game.directionY = -1
            game.directionX = 0
        elif(answerIndex == 1):
            game.directionY = 1
            game.directionX = 0
        elif(answerIndex == 2):
            game.directionX = -1
            game.directionY = 0
        elif(answerIndex == 3):
            game.directionX = 1
            game.directionY = 0
        snakeSave = copy.deepcopy(game.snake)
        fruitSave = game.fruit.copy()
        game.update()
        if(fruitSave != game.fruit):
            dataSinceLastFood = []
        else:
            dataSinceLastFood.append({"snake":copy.deepcopy(snakeSave),"index":answerIndex})
        if(game.checkState() == False):
            return "GameOver"
        ui.grid = game.getGrid()
        ui.head = game.snake[-1]
    
    def replayGame():
        global game
        global ui
        game = snakeGame.Game(gameSize)
        ui.grid = game.getGrid()
    
    ui.startGame(game.getGrid(),False,updateGrid=updateGrid,replayGame=replayGame)
        

ui.startChoosingModel(allModel,handleModelChoice)