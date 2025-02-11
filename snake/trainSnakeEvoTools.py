import json
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
import operator
import matplotlib.pyplot as plt
import copy
import pickle
import time

SELECTIONSIZE = 5 #The number of snake in survivor

MULTIPLIER = 5 #The number of agent wich will be created at start for each place in the selection (totale agent generated = MULTIPLIER * SELECTIONSIZE)

NEARHEAD = False

ITERATION = 50

LEARNINGRATE = 0.05

class trainSnakeEvo():
    def __init__(self, gameSize: int, averageAim: int, hiddenLayers: list[int] = [], activationFunction: callable = None, neuroneActivation: list = None) -> None:
        if(NEARHEAD):
            self.firstNeurones = (((gameSize*2)-1)**2-1)*2
        else:
            self.firstNeurones =  gameSize**2*3
        self.gameSize = gameSize
        self.averageAim = averageAim
        self.childPerformance = []
        self.child = [classe.FNN([self.firstNeurones]+hiddenLayers+[4],learningRate=LEARNINGRATE,neuroneActivation=neuroneActivation,batch_size=1) for i in range(SELECTIONSIZE*MULTIPLIER)]
        self.parents = []
        self.hiddenLayers = hiddenLayers
        self.activationFunction = activationFunction
        self.neuroneActivation = neuroneActivation
        self.iterationData = []
    
    def train(self) -> classe.NN:
        currentGeneration = 1
        survivor = []

        startingTime = time.time()
        while(True):
            if(currentGeneration == 1):
                print("currentGeneration : "+str(currentGeneration),end="\r")
            else:
                print("currentGeneration : "+str(currentGeneration)+" maxLength : "+str(round(survivor[0]["performance"],2))+" averageLength : "+str(trainSnakeEvo.getAveragePerformance(survivor))+" elapsed time : "+str(round((time.time()-startingTime)/60,2))+"m")
            if(currentGeneration > 1):
                security = False
                for i in range(len(survivor)):
                    if(survivor[i]["performance"] >= float(self.averageAim)):
                        security = True
                        verifiyPerformance = self.__runChild(survivor[i]["network"],modification=False)
                        if(verifiyPerformance >= float(self.averageAim)):
                            return survivor[i]["network"]
                        else:
                            survivor[i]["performance"] = verifiyPerformance
                    else:
                        break
                if(security):
                    print("The bests agents of the generation ",str(currentGeneration-1)+" failed the security test !")
            self.childPerformance = []
            
            for i in range(len(self.child)):#benchmarking the childs
                self.childPerformance.append({"number" : (currentGeneration-1)*SELECTIONSIZE*MULTIPLIER+i,"performance":self.__runChild(self.child[i]),"network":copy.deepcopy(self.child[i])})
            
            for i in range(len(survivor)):#re-benchmarking the survivor to reduce randomness
                survivor[i]["performance"] = self.__runChild(survivor[i]["network"])
            
            self.childPerformance.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor += self.childPerformance[:SELECTIONSIZE]
            survivor.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor = survivor[:SELECTIONSIZE]
            
            tempNetworks = []
            for i in range(len(survivor)):#creating the next childs
                for x in range(i+1,len(survivor)):
                    tempNetworks.append(classe.FNN([self.firstNeurones]+self.hiddenLayers+[4],learningRate=LEARNINGRATE,neuroneActivation=self.neuroneActivation,parents=[survivor[i]["network"],survivor[x]["network"]],batch_size=1))
            
            self.child = copy.deepcopy(tempNetworks)
            self.iterationData.append({"generation":currentGeneration,"maxLength":survivor[0]["performance"],"averageLength":trainSnakeEvo.getAveragePerformance(survivor)})
            plt.plot([i+1 for i in range(len(self.iterationData))],[self.iterationData[i]["averageLength"] for i in range(len(self.iterationData))], label="AverageLength of the selected agent")
            plt.plot([i+1 for i in range(len(self.iterationData))],[self.iterationData[i]["maxLength"] for i in range(len(self.iterationData))],label = "AverageLength of the best agent")
            plt.xlabel("generation")
            plt.ylabel("snake length")
            #plt.savefig('evoData.png')
            currentGeneration += 1

    def getAveragePerformance(survivor : list) -> float:
        somme = 0
        for i in survivor:
            somme += i["performance"]
        return round(somme/len(survivor),2)

    def __runChild(self,network : classe.NN,modification = True) -> float:#return the average length of the agent on a certain number of simulation
        currentPerformance = []
        for i in range(ITERATION):
            state = None
            game = snakeGame.Game(self.gameSize)
            dataSincelastFood = []
            previousDataSinceLastFood = []
            while(state == None):
                Networkinput = trainSnakeEvo.generateInput(game.getGrid(),True,game.snake)
                result = network.forward(np.array([np.array(Networkinput)]))[0]
                supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,game.snake,result.tolist(),dataSincelastFood,game)
                errors = [0]*4
                '''for i in range(4):
                    if(supervisedResult[i] < 0):
                        errors[i] = -result[i]'''
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
                snakeSave = copy.deepcopy(game.snake)
                game.update()
                state = game.checkState()
                if(game.fruit != fruitSave):
                    if(modification):
                        errors[answerIndex] = 1-result[answerIndex]
                        network.backward([errors])
                    previousDataSinceLastFood = copy.deepcopy(dataSincelastFood)
                    dataSincelastFood = []
                else:
                    dataSincelastFood.append({"snake":copy.deepcopy(snakeSave),"index":answerIndex,"grid":game.getGrid()})
                    if(len(dataSincelastFood) > game.size**2):
                        state = False
                    if(modification):
                        if(state == False and False):
                            errors[answerIndex] = -result[answerIndex]
                            network.backward(errors)
                            for i in range(len(previousDataSinceLastFood)):
                                Networkinput = trainSnakeEvo.generateInput(previousDataSinceLastFood[i]["grid"],True,previousDataSinceLastFood[i]["snake"])
                                rawResult = network.forward(np.array(Networkinput))
                                supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,previousDataSinceLastFood[i]["snake"],rawResult,previousDataSinceLastFood[:i],game)
                                answerIndex = previousDataSinceLastFood[i]["index"]
                                errors = [0]*4
                                if(supervisedResult[answerIndex] > 0):
                                    errors[answerIndex] = -rawResult[answerIndex] * (0.5+(i/(len(previousDataSinceLastFood)*2)))
                                    network.backward(errors)
                            for i in range(len(dataSincelastFood)):
                                Networkinput = trainSnakeEvo.generateInput(dataSincelastFood[i]["grid"],True,dataSincelastFood[i]["snake"])
                                rawResult = network.forward(np.array(Networkinput))
                                supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,dataSincelastFood[i]["snake"],rawResult,dataSincelastFood[:i],game)
                                answerIndex = dataSincelastFood[i]["index"]
                                errors = [0]*4
                                if(supervisedResult[answerIndex] > 0):
                                    errors[answerIndex] = -rawResult[answerIndex] * (0.5+(i/(len(dataSincelastFood)*2)))
                                    network.backward(errors)
                        else:#Reward the snake when it get closer to the fruit
                            currentDistance = abs(game.snake[-1][0]-game.fruit[0])+abs(game.snake[-1][1]-game.fruit[1])
                            previousDistance = abs(headSave[0]-game.fruit[0])+abs(headSave[1]-game.fruit[1])
                            if(currentDistance < previousDistance):
                                errors[answerIndex] = (1-result[answerIndex]) * 0.2
                                network.backward([errors])
                            elif(currentDistance > previousDistance):
                                errors[answerIndex] = -result[answerIndex] * 0.2
                                network.backward([errors])

            currentPerformance.append(len(game.snake))
        return sum(currentPerformance)/len(currentPerformance)
    
    def superviseAnswer(gameSize : int,snake : list[list[int]],result : list[float],dataSinceLastFood : list[dict],game : snakeGame.Game) -> int:
        modifiedResult = copy.deepcopy(result)
        deadEnd = trainSnakeEvo.checkDeadEnd(game,snake)
        if(snake[-1][1]-1 < 0 or ([snake[-1][0],snake[-1][1]-1] in snake[1:])):
            modifiedResult[0] = -1
        elif (not deadEnd[2]):
            modifiedResult[0] = -0.6

        if(snake[-1][1]+1 >= gameSize or ([snake[-1][0],snake[-1][1]+1] in snake[1:])):
            modifiedResult[1] = -1
        elif(not deadEnd[3]):
            modifiedResult[1] = -0.6
        
        if(snake[-1][0]-1 < 0 or ([snake[-1][0]-1,snake[-1][1]] in snake[1:])):
            modifiedResult[2] = -1
        elif(not deadEnd[0]):
            modifiedResult[2] = -0.6
        
        if(snake[-1][0]+1 >= gameSize or ([snake[-1][0]+1,snake[-1][1]] in snake[1:])):
            modifiedResult[3] = -1
        elif(not deadEnd[1]):
            modifiedResult[3] = -0.6
        
        for i in range(len(dataSinceLastFood)):
            if(dataSinceLastFood[i]["snake"] == snake):
                modifiedResult[dataSinceLastFood[i]["index"]] = -0.5
        return modifiedResult

    def checkDeadEnd(game : snakeGame.Game,snake : list[list[int]]) -> list[bool]:
        def getZoneSize(case : list[int],game : snakeGame.Game):
            previous = []
            current = [case]
            grid = game.getGrid()
            while(len(previous) != len(current)):
                previous = copy.deepcopy(current)
                for case in current:
                    for i in range(-1,2,2):
                        if(case[0]+i >= 0 and case[0]+i < len(grid)):
                            if(grid[case[1]][case[0]+i] != -1):
                                if([case[0]+i,case[1]] not in current):
                                    current.append([case[0]+i,case[1]])
                    for j in range(-1,2,2):
                        if(case[1]+j >= 0 and case[1]+j < len(grid)):
                            if(grid[case[1]+j][case[0]] != -1):
                                if([case[0],case[1]+j] not in current):
                                    current.append([case[0],case[1]+j])
            size = len(current)
            game = copy.deepcopy(game)
            if(size >= len(game.snake)):
                return size
            else:
                game.snake = game.snake[size:]
                previous = []
                current = [case]
                grid = game.getGrid()
                while(len(previous) != len(current)):
                    previous = copy.deepcopy(current)
                    for case in current:
                        for i in range(-1,2,2):
                            if(case[0]+i >= 0 and case[0]+i < len(grid)):
                                if(grid[case[1]][case[0]+i] != -1):
                                    if([case[0]+i,case[1]] not in current):
                                        current.append([case[0]+i,case[1]])
                        for j in range(-1,2,2):
                            if(case[1]+j >= 0 and case[1]+j < len(grid)):
                                if(grid[case[1]+j][case[0]] != -1):
                                    if([case[0],case[1]+j] not in current):
                                        current.append([case[0],case[1]+j])
                    size = len(current)
                    return size

        data = []
        result = [True]*4
        game = copy.deepcopy(game)
        game.snake = game.snake[1:]
        head = game.snake[-1]
        grid = game.getGrid()
        for i in range(-1,2,2):
            if(head[0]+i >= 0 and head[0]+i < len(grid)):
                if(grid[head[1]][head[0]+i] != -1):
                    data.append(getZoneSize([head[0]+i,head[1]],game))
                else:
                    data.append(-1)
            else:
                data.append(-1)
        for j in range(-1,2,2):
            if(head[1]+j >= 0 and head[1]+j < len(grid)):
                if(grid[head[1]+j][head[0]] != -1):
                    data.append(getZoneSize([head[0],head[1]+j],game))
                else:
                    data.append(-1)
            else:
                data.append(-1)
        for i in range(4):
            if(data[i] < len(snake)):
                result[i] = False
        if(result.count(False) == 4):
            result[data.index(max(data))] = True
        return result

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
    
    def generateInput(grid : list[list[int]],seeAllMap : bool,snake : list[list[int]]) -> list[int]:
        networkInput = []
        snakeHead = snake[-1]
        gameSize = len(grid)
        if(NEARHEAD):
            radius = 1
            if(seeAllMap):
                radius = gameSize-1
            for j in range(2):
                #If j == 0 we are searching for food
                #If j == 1 we are searching for danger
                for i in range(-radius,radius+1):
                    for x in range(-radius,radius+1):
                        if(i != 0 or x != 0):
                            if((snakeHead[1]+i >= 0 and snakeHead[0]+x >= 0 and snakeHead[1]+i < gameSize and snakeHead[0]+x < gameSize)):#if the case is in the grid
                                if(j == 0):
                                    if(grid[snakeHead[1]+i][snakeHead[0]+x] == 1):
                                        networkInput.append(1)
                                    else:
                                        networkInput.append(0)
                                elif(j == 1):
                                    if(grid[snakeHead[1]+i][snakeHead[0]+x] == -1):
                                        networkInput.append(1)
                                    else:
                                        networkInput.append(0)
                            elif j == 1:networkInput.append(1)
                            else:networkInput.append(0)
        else:
            for j in range(3):
                #If j == 0 we are searching for the head
                #If j == 1 we are searching for food
                #If = -- 2 we are searching for snake body
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
        answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(trainSnakeEvo.superviseAnswer(game.size,game.snake,result.tolist(),copy.deepcopy(dataSinceLastFood),game)))

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