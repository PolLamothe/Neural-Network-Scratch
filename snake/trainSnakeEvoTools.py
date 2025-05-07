import json
import math
import os

import numpy as np
if(os.getcwd().split(os.path.sep)[-1] != "snake"):
    import snake.snakeGame as snakeGame
else:
    import snakeGame as snakeGame
import sys
sys.path.append("../")
import classe
import random
import copy
import pickle
import time

MEAN_SIZE = 500

WIN_SIZE = 5000

class trainSnakeEvo():
    def __init__(self, gameSize: int, averageAim: int, network : classe.CNN) -> None:
        self.gameSize = gameSize
        self.averageAim = averageAim
        self.childPerformance = []
        self.network = network
        self.iterationData = []
    
    def train(self) -> classe.NN:
        startingTime = time.time()
        winnedGames = []

        lastPerformance = [0 for i in range(MEAN_SIZE)]
        count = 0
        maxAverage = 0

        while(sum(lastPerformance)/len(lastPerformance) < self.averageAim):
            if(sum(lastPerformance)/len(lastPerformance) > maxAverage):
                maxAverage = round(sum(lastPerformance)/len(lastPerformance),2)
            print(count,round(sum(lastPerformance)/len(lastPerformance),2),"max average reached : ",maxAverage," time elapsed : ",int((time.time()-startingTime)/60)," min ","  ",end="\r")
            state = None
            game = snakeGame.Game(self.gameSize)
            previousData = []
            moveSinceLastFruit = 0
            while(state == None):
                Networkinput = trainSnakeEvo.generateInput(game.getGrid(),game.snake)
                result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,game.snake,result.tolist(),previousData)
                errors = [0]*4
                for i in range(4):
                    if(supervisedResult[i] == -1):
                        errors[i] = -result[i]
                answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(supervisedResult))

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
                
                fruitSave = game.fruit.copy()
                headSave = game.snake[-1].copy()
                game.update()
                state = game.checkState()
                previousData.append({"snake":copy.deepcopy(game.snake),"index":answerIndex,"fruit":copy.deepcopy(game.fruit),"original" : True,"forbidden" : None})
                if(game.fruit != fruitSave):
                    errors[answerIndex] = 1-result[answerIndex]
                    self.network.backward([errors])
                    moveSinceLastFruit = 0
                else:
                    moveSinceLastFruit += 1
                    currentDistance = abs(game.snake[-1][0]-game.fruit[0])+abs(game.snake[-1][1]-game.fruit[1])
                    previousDistance = abs(headSave[0]-game.fruit[0])+abs(headSave[1]-game.fruit[1])
                    if(currentDistance < previousDistance):
                        errors[answerIndex] = (1-result[answerIndex]) * max(0.5-(len(game.snake)/(self.gameSize**2+self.gameSize)),0)
                        self.network.backward(np.array([errors]))
                    elif(currentDistance > previousDistance):
                        errors[answerIndex] = -result[answerIndex] * max(0.5-(len(game.snake)/(self.gameSize**2 + self.gameSize)),0)
                        self.network.backward(np.array([errors]))
                    if(moveSinceLastFruit > game.size**3):
                        state = False
                    elif(state == False):
                        previousData.pop()
                        game.snake = previousData[-1]["snake"]
                        game.fruit = previousData[-1]["fruit"]
                        possibility = trainSnakeEvo.exploreEveryPossibility(game,previousData,len(previousData),True)
                        filtred = list(filter(lambda data : data["original"] == True,possibility))
                        
                        index = (possibility.index(filtred[-1]))
                        recommandedIndex = possibility[index+1]["index"]
                        bannedIndex = previousData[index+1]["index"]
                        
                        tempGame = snakeGame.Game(self.gameSize)
                        tempGame.snake = filtred[-1]["snake"]
                        tempGame.fruit = filtred[-1]["fruit"]
                        
                        Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                        result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                        
                        error = [0 for i in range(4)]
                        error[recommandedIndex] = (1-result[recommandedIndex])
                        error[bannedIndex] = -result[bannedIndex]
                        self.network.backward(np.array([error]))
                           
                if(state == True or len(game.snake) == self.gameSize**2-1):
                    if(len(winnedGames) > WIN_SIZE):
                        winnedGames.pop(0)
                    elif(len(winnedGames) == WIN_SIZE):
                        print("The winned game variable is full !")
                    winnedGames.append(copy.deepcopy(previousData))
                    for winnedGame in winnedGames:
                        inputs = []

                        indexAboveMean = None
                        for (index,data) in enumerate(winnedGame):
                            if(len(data["snake"]) > sum(lastPerformance)/len(lastPerformance)):
                                indexAboveMean = index
                                break

                        for i in range(indexAboveMean,len(winnedGame)):
                            game = snakeGame.Game(self.gameSize)
                            game.snake = winnedGame[i-1]["snake"]
                            game.fruit = winnedGame[i-1]["fruit"]

                            Networkinput = trainSnakeEvo.generateInput(game.getGrid(),game.snake)
                            inputs.append(copy.deepcopy(Networkinput))
                        
                        averageError = 1
                        while (averageError > .1):
                            averageError = 0
                            errors = []
                            
                            result = self.network.forward(np.array(inputs))

                            for i in range(indexAboveMean,len(winnedGame)):
                                error = [-result[i-indexAboveMean][j]*0.1 for j in range(4)]
                                error[winnedGame[i]["index"]] = 1-result[i-indexAboveMean][winnedGame[i]["index"]]
                                averageError += (1-result[i-indexAboveMean][winnedGame[i]["index"]])/len(result)
                                errors.append(copy.deepcopy(error))
                            
                            self.network.backward(np.array(errors))
                    state = True
            count += 1
            lastPerformance.append(len(game.snake))
            lastPerformance.pop(0)
        print("\nThe training is over !")
        trainSnakeEvo.benchmarkModel(self.network,self.gameSize)
        return self.network
    
    def benchmarkModel(network : classe.NN,gameSize : int):
        lengthHistory = []
        for i in range(1000):
            print(i/10,"%",end="\r")
            state = None
            game = snakeGame.Game(gameSize)
            moveSinceLastFruit = 0
            previousData = []
            while(state == None):
                Networkinput = trainSnakeEvo.generateInput(game.getGrid(),game.snake)
                result = network.forward(np.array([np.array(Networkinput)]))[0]
                supervisedResult = trainSnakeEvo.superviseAnswer(gameSize,game.snake,result.tolist(),previousData)
                answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(supervisedResult))

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
                                
                fruitSave = game.fruit.copy()
                game.update()
                state = game.checkState()
                previousData.append({"snake":copy.deepcopy(game.snake),"index":answerIndex,"fruit":copy.deepcopy(game.fruit),"original" : True,"forbidden" : None})

                if(len(game.snake) == gameSize**2-1):
                    state = True
                
                if(game.fruit != fruitSave):
                    moveSinceLastFruit = 0
                else:
                    moveSinceLastFruit += 1
                
                if(moveSinceLastFruit > gameSize**3):
                    state = False

            lengthHistory.append(len(copy.deepcopy(game.snake)))
        print("The agent achieve an average length of ",round(sum(lengthHistory)/len(lengthHistory),2)," on ",MEAN_SIZE*2," games")
    def exploreEveryPossibility(game : snakeGame.Game,previousData : list[dict],lengthToBeat : int,canRewind : bool) -> int:
        
        result = [1 for i in range(4)]
        supervisedResult = trainSnakeEvo.superviseAnswer(len(game.getGrid()),game.snake,result,previousData)
        
        if(previousData[-1]["forbidden"] != None):
            supervisedResult[previousData[-1]["forbidden"]] = -1
        
        if(len(previousData) > lengthToBeat):
            return previousData

        for i in range(4):
            if(supervisedResult[i] > 0 ):
                gameCopy = copy.deepcopy(game)
                if(i == 0):
                    gameCopy.directionY = -1
                    gameCopy.directionX = 0
                elif(i == 1):
                    gameCopy.directionY = 1
                    gameCopy.directionX = 0
                elif(i == 2):
                    gameCopy.directionX = -1
                    gameCopy.directionY = 0
                elif(i == 3):
                    gameCopy.directionX = 1
                    gameCopy.directionY = 0
        
                gameCopy.update()

                previousDataCopy = copy.deepcopy(previousData)
                previousDataCopy.append({"snake":gameCopy.snake,"index":i,"fruit":gameCopy.fruit,"forbidden" : None,"original" : False})
                result = trainSnakeEvo.exploreEveryPossibility(gameCopy,previousDataCopy,lengthToBeat,False)
                if(result != None):
                    return copy.deepcopy(result)

        if(canRewind):
            gameCopy = copy.deepcopy(game)
            previousDataCopy = copy.deepcopy(previousData)
            previousDataCopy.pop()
            previousDataCopy[-1]["forbidden"] = previousData[-1]["index"]
            gameCopy.snake = previousDataCopy[-1]["snake"]
            gameCopy.fruit = previousDataCopy[-1]["fruit"]
            return trainSnakeEvo.exploreEveryPossibility(gameCopy,previousDataCopy,lengthToBeat,True)
        else:
            return None
    
    def superviseAnswer(gameSize : int,snake : list[list[int]],result : list[float],previousData : list[dict]) -> int:
        modifiedResult = copy.deepcopy(result)
        if(snake[-1][1]-1 < 0 or ([snake[-1][0],snake[-1][1]-1] in snake[1:])):
            modifiedResult[0] = -1

        if(snake[-1][1]+1 >= gameSize or ([snake[-1][0],snake[-1][1]+1] in snake[1:])):
            modifiedResult[1] = -1
        
        if(snake[-1][0]-1 < 0 or ([snake[-1][0]-1,snake[-1][1]] in snake[1:])):
            modifiedResult[2] = -1
        
        if(snake[-1][0]+1 >= gameSize or ([snake[-1][0]+1,snake[-1][1]] in snake[1:])):
            modifiedResult[3] = -1
        
        for i in range(len(previousData)):
            if(previousData[i]["snake"] == snake):
                modifiedResult[previousData[i]["index"]] = -0.5
        return modifiedResult

    def getAllMaxIndex(answer : list[float]) -> list[int]:
        max = None
        result = []
        for i in range(len(answer)):
            if(max == None):
                max = answer[i]
                result.append(i)
            else:
                if(answer[i] > max):
                    max = answer[i]
                    result = []
                    result.append(i)
                elif(answer[i] == max):
                    result.append(i)
        return result
    
    def generateInput(grid : list[list[int]],snake : list[list[int]]) -> list[int]:
        networkInput = []
        gameSize = len(grid)
        for j in range(3):
            #If j == 0 we are searching for the head
            #If j == 1 we are searching for food
            #If j == 2 we are searching for snake body
            for i in range(gameSize):
                for x in range(gameSize):
                    if(j == 0):
                        if([i,x] == snake[-1]):
                            networkInput.append(1)
                        else:
                            networkInput.append(0)
                    elif(j == 1):
                        if(grid[i][x] == 1):
                            networkInput.append(1)
                        else:
                            networkInput.append(0)
                    else:
                        if(grid[i][x] == -1):
                            networkInput.append(1)
                        else:
                            networkInput.append(0)
        return networkInput

def getWholeGameData() -> list[dict]:
    allModel = []
    for (dirpath, dirnames, filenames) in os.walk("../snake/model"):
        allModel.extend(filenames)

    for model in allModel:
        if(model.split(".")[1] != "pkl"): #keeping only the model files
            allModel.remove(model)
        else:
            model = model.split("_")[2]

    bestModel = None

    with open("../snake/model/trainedData.json","r") as file:
        jsonData = json.loads(file.read())
        for i in range(len(allModel)):
            if(bestModel == None or jsonData[bestModel]["aim"] < jsonData[allModel[i].split("_")[1]]["aim"]):
                bestModel = allModel[i].split("_")[1]

    with open("../snake/model/snake_"+bestModel+"_.pkl", "rb") as file:
        network : classe.NN = pickle.load(file)
    
    game = snakeGame.Game(5)
    dataSinceLastFood = []
    data = []

    while(game.checkState() == None):
        data.append({
            "fruit":copy.deepcopy(game.fruit),
            "snake":copy.deepcopy(game.snake)
        })
        result = network.forward(np.array([trainSnakeEvo.generateInput(game.getGrid(),True,game.snake)]))[0]
        answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(trainSnakeEvo.superviseAnswer(game.size,game.snake,result.tolist(),copy.deepcopy(dataSinceLastFood))))

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
    return data