import json
import snakeGame
import threading
import numpy as np
import random
import os

import sys
sys.path.append("../")
import classe

class snakeTrainTools():

    def __init__(self,gameSize : int, seeAllMap : bool,averageAim : int,hiddenLayers : list[int]=[],activationFunction : callable=classe.sigmoid,softmaxState : bool = False) -> None:
        self.state = None
        self.previousGrid = []#store every step of the current game
        self.stepSinceLastFood = 0
        self.previousLength : list[int] = [4]
        self.dataSinceLastFood : list[dict] = []#data stored like : {grid,head,answerIndex,answerValue}
        self.iteration = 0
        self.gameSize = gameSize
        self.seeAllMap = seeAllMap
        self.previousHead = None
        self.fileName = "./data/game_"+str(gameSize)
        self.averageAim = averageAim
        if(seeAllMap):self.fileName+="_FullMap"
        else:self.fileName+="_NearHead"
        self.fileName+=".json"

        if(seeAllMap == False):
            self.network = classe.Networks([8*2]+hiddenLayers+[4],activationFunction,learningRate=0.1,softmaxState=softmaxState)
        else:
            #self.network = classe.Networks([(((gameSize*2)-1)**2-1)*2]+hiddenLayers+[4],classe.sigmoid,0.1)
            self.network = classe.Networks([gameSize**2*3]+hiddenLayers+[4],activationFunction,learningRate=0.1,softmaxState=softmaxState)

    def train(self):
        with open(self.fileName, 'w') as json_file:
            json.dump({"gameSize":self.gameSize,"data" : []},json_file)

        self.game = snakeGame.Game(self.gameSize)
        maxlength = len(self.game.snake)

        thread = threading.Thread(target=self.__checkIA)
        thread.start()

        while(sum(self.previousLength)/len(self.previousLength) < self.averageAim):
            if(self.iteration < 99):
                print("number of iteration :"+str(self.iteration)+" max length : "+str(maxlength),end="\r")
            else:
                print("number of iteration :"+str(self.iteration)+" max length : "+str(maxlength)+" average length : "+str(sum(self.previousLength)/len(self.previousLength)),end="\r")

            Networkinput = snakeTrainTools.generateInput(self.game.getGrid(),self.seeAllMap,self.game.snake)
                
            result = self.network.forward(Networkinput)

            answerIndex = random.choice(snakeTrainTools.getAllMaxIndex(result))

            if(answerIndex == 0):
                self.game.directionY = -1
                self.game.directionX = 0
            elif(answerIndex == 1):
                self.game.directionY = 1
                self.game.directionX = 0
            elif(answerIndex == 2):
                self.game.directionX = -1
                self.game.directionY = 0
            elif(answerIndex == 3):
                self.game.directionX = 1
                self.game.directionY = 0
            
            fruitSave = self.game.fruit.copy()
            self.game.update()
            self.previousGrid.append({"head":self.game.snake[-1],"data":self.game.getGrid()})

            if(self.game.fruit != fruitSave):#if the snake ate a fruit
                errors = [0]*4
                for i in range(len(errors)):
                    errors[i] = -result[i]
                errors[answerIndex] = 1-result[answerIndex]
                self.network.backward(errors)
                self.previousHead = None
                self.stepSinceLastFood = 0
                self.dataSinceLastFood = []

            state = self.game.checkState()
            if(state == False or self.stepSinceLastFood > self.gameSize**2*2):#If we have lost
                errors = [0]*4
                for i in range(len(errors)):
                    errors[i] = 1-result[i]
                errors[answerIndex] = -result[answerIndex]
                self.network.backward(errors)
                '''if(self.seeAllMap):
                    for step in self.dataSinceLastFood:
                        errors = [0]*4
                        errors[step["answerIndex"]] = -step["answerValue"]*0.2
                        self.network.backward(errors,snakeTrainTools.generateInput(step["grid"],True,step["snake"]))'''

                if(len(self.game.snake) > maxlength):
                    maxlength = len(self.game.snake)
                    self.__addToFile()
                self.__reset()
            elif(state == True):
                if(len(self.game.snake) > maxlength):
                    maxlength = len(self.game.snake)
                    self.__addToFile()
                errors = [0]*4
                errors[answerIndex] = (1-result[answerIndex])
                self.network.backward(errors)

                if(self.seeAllMap):
                    for step in self.dataSinceLastFood:
                        errors = [0]*4
                        errors[step["answerIndex"]] = (1-result[answerIndex])*0.25
                        self.network.backward(errors,snakeTrainTools.generateInput(step["grid"],True,step["snake"]))

                self.__reset()
            elif(self.seeAllMap):#The network can learn to move to the food only if he can see where it is
                if(self.previousHead != None):
                    currentDistance = abs(self.game.snake[-1][0]-self.game.fruit[0])+abs(self.game.snake[-1][1]-self.game.fruit[1])
                    previousDistance = abs(self.previousHead[0]-self.game.fruit[0])+abs(self.previousHead[1]-self.game.fruit[1])
                    if(currentDistance < previousDistance):
                        errors = [0]*4
                        errors[answerIndex] = (1-result[answerIndex])*0.5
                        self.network.backward(errors)
            if(state == None):
                self.dataSinceLastFood.append({"grid":self.game.getGrid(),"snake":self.game.snake,"answerIndex":answerIndex,"answerValue":result[answerIndex]})
                self.previousHead = self.game.snake[-1]
                self.stepSinceLastFood += 1
            if(state == None and False):#reward for surviving
                errors = [0]*4
                errors[answerIndex] = (1-result[answerIndex])*0.1
                self.network.backward(errors)
        print("\nTraining is over !")

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

    def __addToFile(self):
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        with open(self.fileName, 'r') as fichier:
                jsonData = json.load(fichier)
        with open(self.fileName, 'w') as json_file:
            jsonData["data"].append({"iteration":self.iteration,"data":self.previousGrid})
            json.dump(jsonData,json_file,cls=NumpyArrayEncoder,indent=4)

    def __reset(self):
        self.iteration += 1
        self.previousLength.append(len(self.game.snake))
        if(self.iteration > 99):
            self.previousLength.pop(0)
        self.game = snakeGame.Game(self.gameSize)
        self.previousGrid = []
        self.previousHead = None
        self.stepSinceLastFood = 0
        self.dataSinceLastFood = []

    def generateInput(grid : list[list[int]],seeAllMap : bool,snake : list[list[int]]):
        networkInput = []
        snakeHead = snake[-1]
        gameSize = len(grid)
        if(not seeAllMap):
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

    def __checkIA(self):
        while(True):
            input("")
            print("\n"+str([round(i,2) for i in self.network.forward(snakeTrainTools.generateInput(self.game.getGrid(),self.seeAllMap,self.game.snake))])+"\n")