import json
import snakeGame
import threading
import numpy as np

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

        if(seeAllMap == False):
            self.network = classe.Networks([8*2]+hiddenLayers+[4],classe.sigmoid,0.1)
        else:
            self.network = classe.Networks([(((gameSize*2)-1)**2-1)*2]+hiddenLayers+[4],classe.sigmoid,0.1)

    def train(self):
        with open('./data/game.json', 'w') as json_file:
            json.dump([],json_file)

        self.game = snakeGame.Game(self.gameSize)
        maxlength = len(self.game.snake)

        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        thread = threading.Thread(target=self.checkIA)
        thread.start()

        while(True):
            self.previousGrid.append(self.game.getGrid())
            print("number of iteration :"+str(self.iteration)+" max length : "+str(maxlength),end="\r")

            Networkinput = self.generateInput()
            result = self.network.forward(Networkinput)

            self.previousResult.append(result)
            self.previousData.append(Networkinput)
            
            answerIndex = np.argmax(self.previousResult[len(self.previousResult)-1])

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
            if(self.game.fruit != fruitSave):
                errors = [0]*4
                for i in range(len(result)):
                    errors[i] = -result[i]
                errors[answerIndex] = 1#-result[answerIndex]
                self.network.backward(errors)
                self.previousData = []
                self.previousResult = []

            state = self.game.checkState()
            if(state == False or len(self.previousData) >= 20):
                errors = [0]*4
                errors[answerIndex] = -1#0-result[answerIndex]
                self.network.backward(errors)
                
                if(len(self.game.snake) > maxlength):
                    with open('./data/game.json', 'r') as fichier:
                        jsonData = json.load(fichier)
                    with open('./data/game.json', 'w') as json_file:
                        jsonData.append({"iteration":self.iteration,"data":self.previousGrid})
                        json.dump(jsonData,json_file,cls=NumpyArrayEncoder,indent=4)

                if(len(self.game.snake) > maxlength):
                    maxlength = len(self.game.snake)
                self.reset()
            elif(state == True):
                with open('./data/game.json', 'r') as fichier:
                        jsonData = json.load(fichier)
                with open('./data/game.json', 'w') as json_file:
                    jsonData.append({"iteration":self.iteration,"data":self.previousGrid})
                    json.dump(jsonData,json_file,cls=NumpyArrayEncoder,indent=4)
                print("won")
                break

    def reset(self):
        self.iteration += 1
        self.game = snakeGame.Game(self.gameSize)
        self.previousData = []
        self.previousResult = []
        self.previousGrid = []

    def generateInput(self):
        networkInput = []
        grid = self.game.getGrid()
        snakeHead = self.game.snake[len(self.game.snake)-1]
        radius = 1
        if(self.seeAllMap):
            radius = self.gameSize-1
        for j in range(2):
            for i in range(-radius,radius+1):
                for x in range(-radius,radius+1):
                    if(i != 0 or x != 0):
                        if((snakeHead[1]+i >= 0 and snakeHead[0]+x >= 0 and snakeHead[1]+i < self.game.size and snakeHead[0]+x < self.game.size)):
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

    def checkIA(self):
        while(True):
            input("")
            print(self.network.forward(self.generateInput()))