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

    def __init__(self,gameSize : int, seeAllMap : bool,hiddenLayers : list[int]=[]) -> None:
        self.state = None
        self.previousData = []#store every input of the network since the last backward propagation
        self.previousResult = []#store every output of the network since the last backward propagation
        self.previousGrid = []#store every step of the current game
        self.iteration = 0
        self.gameLength = 0
        self.gameSize = gameSize
        self.seeAllMap = seeAllMap
        self.previousHead = None
        self.fileName = "./data/game_"+str(gameSize)
        if(seeAllMap):self.fileName+="_FullMap"
        else:self.fileName+="_NearHead"
        self.fileName+=".json"

        if(seeAllMap == False):
            self.network = classe.Networks([8*2]+hiddenLayers+[4],classe.sigmoid,0.1)
        else:
            self.network = classe.Networks([(((gameSize*2)-1)**2-1)*2]+hiddenLayers+[4],classe.sigmoid,0.1)

    def train(self):
        with open(self.fileName, 'w') as json_file:
            json.dump({"gameSize":self.gameSize,"data" : []},json_file)

        self.game = snakeGame.Game(self.gameSize)
        maxlength = len(self.game.snake)

        thread = threading.Thread(target=self.__checkIA)
        thread.start()

        while(True):
            print("number of iteration :"+str(self.iteration)+" max length : "+str(maxlength),end="\r")

            Networkinput = self.__generateInput()
            result = self.network.forward(Networkinput)

            self.previousData.append(Networkinput)
            
            answerIndex = random.choice(snakeTrainTools.getAllMaxIndex(result))
            self.previousResult.append([answerIndex,result[answerIndex]])

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
            self.previousGrid.append(self.game.getGrid())

            if(self.game.fruit != fruitSave):#if the snake ate a fruit
                if(not self.seeAllMap):
                    errors = [0]*4
                    errors[answerIndex] = 1-result[answerIndex]
                    self.network.backward(errors)
                else:
                    for i in range(len(self.previousData)-1):
                        continue
                        errors = [0]*4
                        errors[self.previousResult[i][0]] = (1-self.previousResult[i][1])/len(self.previousData)
                        self.network.backward(errors,self.previousData[i])
                    errors = [0]*4
                    errors[answerIndex] = 1-result[answerIndex]
                    self.network.backward(errors)
                self.previousData = []
                self.previousResult = []
                self.previousHead = None

            state = self.game.checkState()
            if(state == False or len(self.previousData) > self.gameSize**2*2):#If we have lost
                if(not self.seeAllMap):
                    errors = [0]*4
                    errors[answerIndex] = -result[answerIndex]
                    self.network.backward(errors)
                else:
                    '''for i in range(len(self.previousData)-1):
                        errors = [0]*4
                        errors[self.previousResult[i][0]] = (-self.previousResult[i][1])/len(self.previousData)
                        self.network.backward(errors,self.previousData[i])'''
                    errors = [0]*4
                    errors[answerIndex] = -result[answerIndex]
                    self.network.backward(errors)

                if(len(self.game.snake) > maxlength):
                    maxlength = len(list(filter(self.checkPosition,self.game.snake)))
                    self.__addToFile()
                self.__reset()
            elif(state == True):
                self.__addToFile()
                break
            elif(self.seeAllMap):
                continue
                if(self.previousHead != None):
                    currentDistance = abs(self.game.snake[-1][0]-self.game.fruit[0])+abs(self.game.snake[-1][1]-self.game.fruit[1])
                    previousDistance = abs(self.previousHead[0]-self.game.fruit[0])+abs(self.previousHead[1]-self.game.fruit[1])
                    if(currentDistance < previousDistance):
                        errors = [0]*4
                        errors[answerIndex] = 0.2
                        self.network.backward(errors)
                    elif(currentDistance > previousDistance):
                        continue
                        errors = [0]*4
                        errors[answerIndex] = -0.1
                        self.network.backward(errors)
            self.previousHead = self.game.snake[-1]
        print("\nwon")
        os._exit(0)

    def checkPosition(self,position : list[int]):
        return position[0] >= 0 and position[0] < self.gameSize and position[1] >= 0 and position[1] < self.gameSize 

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
        self.game = snakeGame.Game(self.gameSize)
        self.previousData = []
        self.previousResult = []
        self.previousGrid = []
        self.previousHead = None

    def __generateInput(self):
        networkInput = []
        grid = self.game.getGrid()
        snakeHead = self.game.snake[len(self.game.snake)-1]
        radius = 1
        if(self.seeAllMap):
            radius = self.gameSize-1
        for j in range(2):
            #If j == 0 we are searching for food
            #If j == 1 we are searching for danger
            for i in range(-radius,radius+1):
                for x in range(-radius,radius+1):
                    if(i != 0 or x != 0):
                        if((snakeHead[1]+i >= 0 and snakeHead[0]+x >= 0 and snakeHead[1]+i < self.game.size and snakeHead[0]+x < self.game.size)):#if the case is in the grid
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
        return networkInput

    def __checkIA(self):
        while(True):
            input("")
            print("\n"+str([round(i,2) for i in self.network.forward(self.__generateInput())])+"\n")