import snakeGame
import sys
sys.path.append("../")
import classe
import random
import operator
import matplotlib.pyplot as plt
import copy
import math

SELECTIONSIZE = 10 #The number of snake in survivor

MULTIPLIER = 10 #The number of agent wich will be created at start for each place in the selection (totale agent generated = MULTIPLIER * SELECTIONSIZE)

NEARHEAD = False

class trainSnakeEvo():
    def __init__(self, gameSize: int, averageAim: int, hiddenLayers: list[int] = [], activationFunction: callable = None, neuroneActivation: list = None) -> None:
        if(NEARHEAD):
            self.firstNeurones = (((gameSize*2)-1)**2-1)*2
        else:
            self.firstNeurones =  gameSize**2*3
        self.gameSize = gameSize
        self.averageAim = averageAim
        self.childPerformance = []
        self.child = [classe.Networks([self.firstNeurones]+hiddenLayers+[4],activation=activationFunction,learningRate=0.1,neuroneActivation=neuroneActivation) for i in range(SELECTIONSIZE*MULTIPLIER)]
        self.parents = []
        self.hiddenLayers = hiddenLayers
        self.activationFunction = activationFunction
        self.neuroneActivation = neuroneActivation
        self.iterationData = []
    
    def train(self) -> classe.Networks:
        currentGeneration = 1
        survivor = []

        while(True):
            if(currentGeneration == 1):
                print("currentGeneration : "+str(currentGeneration),end="\r")
            else:
                print("currentGeneration : "+str(currentGeneration)+" maxLength : "+str(survivor[0]["performance"])+" averageLength : "+str(trainSnakeEvo.getAveragePerformance(survivor)),end="\r")
            if(currentGeneration > 1):
                security = False
                for i in range(len(survivor)):
                    if(survivor[i]["performance"] >= float(self.averageAim)):
                        security = True
                        verifiyPerformance = self.__runChild(survivor[i]["network"])
                        if(verifiyPerformance >= float(self.averageAim)):
                            return survivor[i]["network"]
                        else:
                            survivor[i]["performance"] = (survivor[i]["performance"]*survivor[i]["verification"]+verifiyPerformance)/(survivor[i]["verification"]+1)
                            survivor[i]["verification"] += 1
                    else:
                        break
                if(security):
                    print("\nThe bests agents of the generation ",str(currentGeneration-1)+" failed the security test !")
            self.childPerformance = []
            for i in range(len(self.child)):#benchmarking the childs
                self.childPerformance.append({"number" : (currentGeneration-1)*SELECTIONSIZE*MULTIPLIER+i,"performance":self.__runChild(self.child[i]),"network":self.child[i],"verification":1})
            '''for i in range(len(survivor)):#re-benchmarking the survivor to reduce randomness
                survivor[i]["performance"] = (survivor[i]["performance"]*survivor[i]["verification"]+self.__runChild(survivor[i]["network"]))/(survivor[i]["verification"]+1)
                survivor[i]["verification"] += 1'''
            self.childPerformance.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor += self.childPerformance[:SELECTIONSIZE]
            survivor.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor = survivor[:SELECTIONSIZE]
            
            tempNetworks = []
            for i in range(len(survivor)):#creating the next childs
                for x in range(i,len(survivor)):
                    tempNetworks.append(classe.Networks([self.firstNeurones]+self.hiddenLayers+[4],activation=self.activationFunction,learningRate=0.1,neuroneActivation=self.neuroneActivation,parents=[survivor[i]["network"],survivor[x]["network"]]))
            
            self.child = tempNetworks.copy()
            self.iterationData.append({"generation":currentGeneration,"maxLength":survivor[0]["performance"],"averageLength":trainSnakeEvo.getAveragePerformance(survivor)})
            plt.plot([i+1 for i in range(len(self.iterationData))],[self.iterationData[i]["averageLength"] for i in range(len(self.iterationData))], label="AverageLength of the selected agent")
            plt.plot([i+1 for i in range(len(self.iterationData))],[self.iterationData[i]["maxLength"] for i in range(len(self.iterationData))],label = "AverageLength of the best agent")
            plt.xlabel("generation")
            plt.ylabel("snake length")
            plt.savefig('evoData.png')
            currentGeneration += 1

    def getAveragePerformance(survivor : list) -> float:
        somme = 0
        for i in survivor:
            somme += i["performance"]
        return round(somme/len(survivor),2)

    def __runChild(self,network : classe.Networks) -> float:#return the average length of the agent on a certain number of simulation
        currentPerformance = []
        for i in range(30):
            state = None
            game = snakeGame.Game(self.gameSize)
            dataSincelastFood = []
            while(state == None):
                Networkinput = trainSnakeEvo.generateInput(game.getGrid(),True,game.snake)
                result = network.forward(Networkinput)
                answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(trainSnakeEvo.superviseAnswer(self.gameSize,game.snake,result,dataSincelastFood)))

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
                    dataSincelastFood = []
                    errors = [0]*4
                    errors[answerIndex] = 1-result[answerIndex]
                    network.backward(errors)
                else:
                    dataSincelastFood.append({"snake":copy.deepcopy(snakeSave),"index":answerIndex,"grid":game.getGrid()})
                    if(state == False):
                        for element in dataSincelastFood:
                            Networkinput = trainSnakeEvo.generateInput(element["grid"],True,element["snake"])
                            result = network.forward(Networkinput)
                            answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(trainSnakeEvo.superviseAnswer(self.gameSize,game.snake,result,dataSincelastFood)))
                            errors = [0]*4
                            errors[answerIndex] = -result[answerIndex]
                            network.backward(errors)
                    else:
                        currentDistance = abs(game.snake[-1][0]-game.fruit[0])+abs(game.snake[-1][1]-game.fruit[1])
                        previousDistance = abs(headSave[0]-game.fruit[0])+abs(headSave[1]-game.fruit[1])
                        if(currentDistance < previousDistance):
                            errors = [0]*4
                            errors[answerIndex] = (1-result[answerIndex])*0.5
                            network.backward(errors)
                        elif(currentDistance > previousDistance):
                            errors = [0]*4
                            errors[answerIndex] = -result[answerIndex]*0.5
                            network.backward(errors)

            currentPerformance.append(len(game.snake))
        return sum(currentPerformance)/len(currentPerformance)
    
    def superviseAnswer(gameSize : int,snake : list[list[int]],result : list[float],dataSinceLastFood : list[dict]) -> int:
        modifiedResult = result.copy()
        errors = [1-i for i in result]
        try:
            if(snake[-1][1]-1 < 0 or ([snake[-1][0],snake[-1][1]-1] in snake[1:])):
                errors[0] = -result[0]
                modifiedResult[0] = -1
            if(snake[-1][1]+1 >= gameSize or ([snake[-1][0],snake[-1][1]+1] in snake[1:])):
                errors[1] = -result[1]
                modifiedResult[1] = -1
            if(snake[-1][0]-1 < 0 or ([snake[-1][0]-1,snake[-1][1]] in snake[1:])):
                errors[2] = -result[2]
                modifiedResult[2] = -1
            if(snake[-1][0]+1 >= gameSize or ([snake[-1][0]+1,snake[-1][1]] in snake[1:])):
                errors[3] = -result[3]
                modifiedResult[3] = -1
            for i in range(len(dataSinceLastFood)):
                if(dataSinceLastFood[i]["snake"] == snake):
                    modifiedResult[dataSinceLastFood[i]["index"]] = -0.5
            #if(network != None):
                #network.backward(errors)
            return modifiedResult
        except IndexError:
            print(snake[-1])

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
