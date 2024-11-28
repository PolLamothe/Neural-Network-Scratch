import snakeGame
from os import walk
import pickle
import snakeTrainTools
import trainSnakeEvoTools
import random
import classe

allModel = []
for (dirpath, dirnames, filenames) in walk("./model"):
    allModel.extend(filenames)

ui = snakeGame.UI()

game = None

def handleModelChoice(name : str):
    global ui
    global game
    with open("./model/"+name, "rb") as file:
        network : classe.Networks = pickle.load(file)
    
    gameSize = int(name.split("_")[1])
    fullMap = name.split("_")[2] == "FullMap.pkl"

    game = snakeGame.Game(gameSize)

    def updateGrid():
        global game
        if(not fullMap):
            result = network.forward(snakeTrainTools.snakeTrainTools.generateInput(game.getGrid(),fullMap,game.snake))
            answerIndex = random.choice(snakeTrainTools.snakeTrainTools.getAllMaxIndex(result))
        else:
            result = network.forward(trainSnakeEvoTools.trainSnakeEvo.generateInput(game.getGrid(),True,game.snake))
            answerIndex = random.choice(trainSnakeEvoTools.trainSnakeEvo.getAllMaxIndex(trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(game.size,game,result,network)))

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
        game.update()
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