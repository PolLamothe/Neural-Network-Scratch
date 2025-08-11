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

MEAN_SIZE = 600

ERROR_SIZE = 5*4

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
        correctedSituations = []

        lastPerformance = [0 for i in range(MEAN_SIZE)]
        count = 0
        maxAverage = 0

        while(self.wheightedAverage(lastPerformance) < self.averageAim):
            if(self.wheightedAverage(lastPerformance) > maxAverage):
                maxAverage = round(self.wheightedAverage(lastPerformance),2)
            print(count,round(self.wheightedAverage(lastPerformance),2),"max average reached : ",maxAverage," winned games : ",len(winnedGames)," time elapsed : ",int((time.time()-startingTime)/60)," min ","  ",end="\r")
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
                zonesNumberSave = getSeparatedZones(game.snake,self.gameSize)
                game.update()
                
                zonesNumber = getSeparatedZones(game.snake,self.gameSize)
                zoneChange = 0
                if(zonesNumberSave < zonesNumber):
                    zoneChange = 1
                    errors[answerIndex] = -result[answerIndex]
                elif(zonesNumberSave > zonesNumber):
                    errors[answerIndex] = 1-result[answerIndex]
                    zoneChange = -1

                state = game.checkState()
                previousData.append({"snake":copy.deepcopy(game.snake),"index":answerIndex,"fruit":copy.deepcopy(game.fruit),"original" : True,"forbidden" : None,"zoneChange" : zoneChange})
                if(game.fruit != fruitSave):
                    errors[answerIndex] = 1-result[answerIndex]
                    self.network.backward([errors])
                    moveSinceLastFruit = 0
                else:
                    moveSinceLastFruit += 1
                    currentDistance = abs(game.snake[-1][0]-game.fruit[0])+abs(game.snake[-1][1]-game.fruit[1])
                    previousDistance = abs(headSave[0]-game.fruit[0])+abs(headSave[1]-game.fruit[1])
                    if(currentDistance < previousDistance):
                        if(errors[answerIndex] == 0):
                            errors[answerIndex] = (1-result[answerIndex]) * ((math.log(1)-math.log(len(game.snake)/(self.gameSize**2-self.gameSize)))/2)
                        #errors[answerIndex] = (1-result[answerIndex]) * max(0.5-(len(game.snake)/(self.gameSize**2)),0)
                    elif(currentDistance > previousDistance):
                        if errors[answerIndex] == 0:
                            errors[answerIndex] = -result[answerIndex] * ((math.log(1)-math.log(len(game.snake)/(self.gameSize**2-self.gameSize)))/2)
                        #errors[answerIndex] = -result[answerIndex] * max(0.5-(len(game.snake)/(self.gameSize**2)),0)
                    self.network.backward(np.array([errors]))

                    if(moveSinceLastFruit > game.size**2*2):
                        state = False
                    elif(state == False):
                        saveIndex = copy.deepcopy(previousData[-1]["index"])
                        previousData.pop()
                        game.snake = previousData[-1]["snake"]
                        game.fruit = previousData[-1]["fruit"]
                        possibility = trainSnakeEvo.exploreEveryPossibility(game,previousData,len(previousData),True)
                        filtred = list(filter(lambda data : data["original"] == True,possibility))
                        
                        index = (possibility.index(filtred[-1]))
                        recommandedIndex = possibility[index+1]["index"]
                        if(index+1 < len(previousData)):
                            bannedIndex = previousData[index+1]["index"]
                        else:
                            bannedIndex = saveIndex

                        rotatedGames = self.rotateGame(filtred[-1]["snake"],filtred[-1]["fruit"])
                        for index,rotatedGame in enumerate(rotatedGames):
                            tempGame = snakeGame.Game(self.gameSize)
                            tempGame.snake = copy.deepcopy(rotatedGame[0])
                            tempGame.fruit = copy.deepcopy(rotatedGame[1])

                            Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                            result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                            supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,tempGame.snake,result,[])

                            error = [0 for i in range(4)]
                            for i in range(4):
                                if(supervisedResult[i] == -1):
                                    error[i] = -result[i]
                            correctedSituations.append({
                                "rotation" : index,
                                "rotatedGame" : copy.deepcopy(rotatedGame),
                                "bannedIndex" : self.get_aligned_answer(index,bannedIndex),
                                "recommandedIndex" : self.get_aligned_answer(index,recommandedIndex)})
                            error[self.get_aligned_answer(index,recommandedIndex)] = (1-result[self.get_aligned_answer(index,recommandedIndex)])
                            error[self.get_aligned_answer(index,bannedIndex)] = -result[self.get_aligned_answer(index,bannedIndex)]
                            self.network.backward(np.array([error]))
                        correctedSituationsSelected = []
                        while(len(correctedSituations) > 0 and len(correctedSituationsSelected) < ERROR_SIZE):
                            correctedSituationsSelected.append(random.choice(correctedSituations))
                        for rotatedGame in correctedSituationsSelected:
                            tempGame = snakeGame.Game(self.gameSize)
                            tempGame.snake = copy.deepcopy(rotatedGame["rotatedGame"][0])
                            tempGame.fruit = copy.deepcopy(rotatedGame["rotatedGame"][1])

                            Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                            result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                            supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,tempGame.snake,result,[])

                            error = [0 for i in range(4)]
                            for i in range(4):
                                if(supervisedResult[i] == -1):
                                    error[i] = -result[i]
                            error[rotatedGame["recommandedIndex"]] = (1-result[rotatedGame["recommandedIndex"]])
                            error[rotatedGame["bannedIndex"]] = -result[rotatedGame["bannedIndex"]]
                            self.network.backward(np.array([error]))
                           
                if(state == True or len(game.snake) == self.gameSize**2-1):
                    winnedGames.append(copy.deepcopy(previousData))
                    state = True
            count += 1
            lastPerformance.append(len(game.snake))
            lastPerformance.pop(0)

            if(random.random() > (1-len(winnedGames)/count)**2 and len(winnedGames) > 0):

                selectedGame = random.choice(winnedGames)

                inputs = [[],[],[],[]]

                indexAboveMean = None
                for (index,data) in enumerate(selectedGame):
                    if(len(data["snake"]) > min(lastPerformance)):
                        indexAboveMean = index
                        break

                for i in range(indexAboveMean,len(selectedGame)):
                    rotatedGames = self.rotateGame(selectedGame[-1]["snake"],selectedGame[-1]["fruit"])

                    for index,rotatedGame in enumerate(rotatedGames):
                        tempGame = snakeGame.Game(self.gameSize)
                        tempGame.snake = copy.deepcopy(rotatedGame[0])
                        tempGame.fruit = copy.deepcopy(rotatedGame[1])

                        Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                        inputs[index].append(copy.deepcopy(Networkinput))

                for x in range(4):
                    errors = []
                    
                    result = self.network.forward(np.array(inputs[x]))

                    for i in range(indexAboveMean,len(selectedGame)):
                        tempGame = snakeGame.Game(self.gameSize)
                        tempGame.snake = selectedGame[i]["snake"]
                        tempGame.fruit = selectedGame[i]["fruit"]
                        
                        index = self.get_aligned_answer(selectedGame[i]["index"],x)
                        supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,tempGame.snake,result[i-indexAboveMean],[])

                        error = [0 for j in range(4)]
                        for j in range(4):
                            if(supervisedResult[j] == -1):
                                error[j] = -result[i-indexAboveMean][j]
                        error[index] = 1-result[i-indexAboveMean][index]
                        if(selectedGame[i]["zoneChange"] == 1):
                            error[index] = -result[i-indexAboveMean][index]
                        errors.append(copy.deepcopy(error))
                    
                self.network.backward(np.array(errors))
        print("\nThe training is over !")
        return copy.deepcopy(self.network)
    
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
                                
                fruitSave = copy.deepcopy(game.fruit)
                game.update()
                state = game.checkState()
                previousData.append({"snake":copy.deepcopy(game.snake),"index":answerIndex,"fruit":copy.deepcopy(game.fruit),"original" : True,"forbidden" : None})

                if(len(game.snake) >= gameSize**2):
                    state = True
                
                if(game.fruit != fruitSave):
                    moveSinceLastFruit = 0
                else:
                    moveSinceLastFruit += 1
                
                if(moveSinceLastFruit > gameSize**3):
                    state = False

            lengthHistory.append(len(copy.deepcopy(game.snake)))
        print("The agent achieve an average length of ",round(sum(lengthHistory)/len(lengthHistory),2)," on ",1000," games")
    
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
    
    def wheightedAverage(self,performances : list[int]) -> float:
        result = 0
        totalCoeff = 0
        for index,performance in enumerate(performances):
            coeff = index/(len(performances)-1)+0.75*((len(performances)-1-index)/(len(performances)-1))
            totalCoeff += coeff
            result += performance * coeff
        return result/totalCoeff
    
    def average(self,performances : list[int ]) -> float:
        return sum(performances)/len(performances)

    def __rotate_point(self,x: int, y: int, alignment: int) -> list[int, int]:
        if alignment == 0:
            return [x, y]
        elif alignment == 1:  # 90°
            return [y, self.gameSize - 1 - x]
        elif alignment == 2:  # 180°
            return [self.gameSize - 1 - x, self.gameSize - 1 - y]
        elif alignment == 3:  # 270°
            return [self.gameSize - 1 - y, x]

    def rotateGame(self,snake : list[list[int]],fruit : list[int]) -> list:
        result = []

        for j in range(4):

            alignedSnake = copy.deepcopy(snake)

            for i in range(len(alignedSnake)):
                alignedSnake[i] = self.__rotate_point(alignedSnake[i][0],alignedSnake[i][1],j)
            
            alignedFruit = self.__rotate_point(fruit[0],fruit[1],j)
            result.append([copy.deepcopy(alignedSnake),copy.deepcopy(alignedFruit)])
        return result
    
    def get_aligned_answer(self,alignment, answerIndex):
        if alignment == 0:
            return answerIndex
        elif alignment == 1:  # Rotation 90°
            return [2, 3, 1, 0][answerIndex]
        elif alignment == 2:  # Rotation 180°
            return [1, 0, 3, 2][answerIndex]
        elif alignment == 3:  # Rotation 270°
            return [3, 2, 0, 1][answerIndex]
    
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
        
        for i in range(len(previousData)-1):
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
        for j in range(4):
            #If j == 0 we are searching for the head
            #If j == 1 we are searching for food
            #If j == 2 we are searching for snake body
            #If j == 3 we are searching for the snake tail
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
                    elif(j == 2):
                        if(grid[i][x] == -1):
                            networkInput.append(1)
                        else:
                            networkInput.append(0)
                    else:
                        if([i,x] == snake[0]):
                            networkInput.append(1)
                        else:
                            networkInput.append(0)
        avaibleSpaces = getAvailableSpaces(snake,snake[-1],len(grid))
        for spaces,head in avaibleSpaces:
            networkInput.append(len(spaces)/(len(grid)**2-4))
            if head:
                networkInput.append(1)
            else:
                networkInput.append(0)
        networkInput.append(len(snake)/(len(grid)**2))
        return networkInput

def getSeparatedZones(snake : list[list[int]],gameSize : int) -> int:
    zones = []
    flatzones = []
    for i in range(gameSize):
        for j in range(gameSize):
            if([i,j] in snake):
                continue
            if(len(flatzones) == gameSize**2-len(snake)):
                return len(zones)
            if([i,j] in flatzones):
                continue
            zone = getZone(snake,flatzones,[i,j],gameSize)
            zones.append(copy.deepcopy(zone))
            for case in zone:
                flatzones.append(copy.deepcopy(case))
    return len(zones)


def getZone(snake : list[list[int]],seenCases : list[list[int]],currentCase : list[int],gameSize : int) -> list[list[int]]:
    if(currentCase[0] >= gameSize or currentCase[0] < 0 or currentCase[1] >= gameSize or currentCase[1] < 0):
        return seenCases
    if(currentCase in snake):
        return seenCases
    if(currentCase in seenCases):
        return seenCases
    copySeenCases = copy.deepcopy(seenCases)
    copySeenCases.append(copy.deepcopy(currentCase))
    copySeenCases = getZone(snake,copySeenCases,[currentCase[0]+1,currentCase[1]],gameSize)
    copySeenCases = getZone(snake,copySeenCases,[currentCase[0]-1,currentCase[1]],gameSize)
    copySeenCases = getZone(snake,copySeenCases,[currentCase[0],currentCase[1]+1],gameSize)
    copySeenCases = getZone(snake,copySeenCases,[currentCase[0],currentCase[1]-1],gameSize)
    return copySeenCases

def getAvailableSpaces(snake : list[list[int]],head : list[int],gameSize : int) -> list[(list | bool)]:
    return [
        getFreeCases(snake,[],[head[0],head[1]-1],gameSize),
        getFreeCases(snake,[],[head[0],head[1]+1],gameSize),
        getFreeCases(snake,[],[head[0]-1,head[1]],gameSize),
        getFreeCases(snake,[],[head[0]+1,head[1]],gameSize),
        ]

def getFreeCases(snake : list[list[int]],seenCases : list[list[int]],currentCase : list[int],gameSize : int) -> (list[list[int]] | bool):
    if(currentCase[0] >= gameSize or currentCase[0] < 0 or currentCase[1] >= gameSize or currentCase[1] < 0):
        return (seenCases,False)
    if(currentCase in snake):
        return (seenCases,False)
    if(currentCase in seenCases):
        return (seenCases,False)
    copySeenCases = copy.deepcopy(seenCases)
    copySeenCases.append(copy.deepcopy(currentCase))
    headState = currentCase == snake[0]
    stateSave = None
    (copySeenCases,stateSave) = getFreeCases(snake,copySeenCases,[currentCase[0]+1,currentCase[1]],gameSize)
    headState = headState or stateSave
    (copySeenCases, stateSave) = getFreeCases(snake,copySeenCases,[currentCase[0]-1,currentCase[1]],gameSize)
    headState = headState or stateSave
    (copySeenCases,stateSave) = getFreeCases(snake,copySeenCases,[currentCase[0],currentCase[1]+1],gameSize)
    headState = headState or stateSave
    (copySeenCases,stateSave) = getFreeCases(snake,copySeenCases,[currentCase[0],currentCase[1]-1],gameSize)
    return (copySeenCases,headState)

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