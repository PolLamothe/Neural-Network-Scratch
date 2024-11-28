import snakeTrainTools
import snakeGame
import classe
import random
import operator
import matplotlib.pyplot as plt

SELECTIONSIZE = 21 #The number of snake in survivor

MULTIPLIER = 10 #The number of snake that will be generated during the first generation

class trainSnakeEvo(snakeTrainTools.snakeTrainTools):
    def __init__(self, gameSize: int, averageAim: int, hiddenLayers: list[int] = [], activationFunction: callable = None, neuroneActivation: list = None) -> None:
        self.gameSize = gameSize
        self.averageAim = averageAim
        self.childPerformance = []
        #self.child = [classe.Networks([gameSize**2*3]+hiddenLayers+[4],activation=activationFunction,learningRate=0.1,neuroneActivation=neuroneActivation) for i in range(SELECTIONSIZE*MULTIPLIER)]
        self.child = [classe.Networks([(((gameSize*2)-1)**2-1)*2]+hiddenLayers+[4],activation=activationFunction,learningRate=0.1,neuroneActivation=neuroneActivation) for i in range(SELECTIONSIZE*MULTIPLIER)]
        self.parents = []
        self.hiddenLayers = hiddenLayers
        self.activationFunction = activationFunction
        self.neuroneActivation = neuroneActivation
        self.iterationData = []
    
    def train(self):
        currentGeneration = 1
        survivor = []

        while(True):
            if(len(survivor) == 0):
                print("currentGeneration : "+str(currentGeneration),end="\r")
            else:
                print("currentGeneration : "+str(currentGeneration)+" maxLength : "+str(survivor[0]["performance"])+" averageLength : "+str(trainSnakeEvo.getAveragePerformance(survivor)),end="\r")
            if(currentGeneration > 1):
                if(survivor[0]["performance"] >= float(self.averageAim)):
                    return survivor[0]["network"]
            self.childPerformance = []
            for i in range(len(self.child)):#benchmarking the childs
                self.childPerformance.append({"number" : (currentGeneration-1)*SELECTIONSIZE*MULTIPLIER+i,"performance":self.__runChild(self.child[i]),"network":self.child[i]})
            self.childPerformance.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor += self.childPerformance[:SELECTIONSIZE]
            survivor.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor = survivor[:SELECTIONSIZE]
            
            tempNetworks = []
            for i in range(1,len(survivor)):#creating the next childs
                for x in range(i,len(survivor)):
                    tempNetworks.append(classe.Networks([self.gameSize**2*3]+self.hiddenLayers+[4],activation=self.activationFunction,learningRate=0.1,neuroneActivation=self.neuroneActivation,parents=[survivor[i]["network"],survivor[x]["network"]]))
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

    def __runChild(self,network : classe.Networks) -> float:
        currentPerformance = []
        for i in range(10):
            state = None
            game = snakeGame.Game(self.gameSize)
            moveSinceLastFood = 0
            while(state == None):
                Networkinput = trainSnakeEvo.generateInput(game.getGrid(),True,game.snake)
                result = network.forward(Networkinput)
                answerIndex = random.choice(snakeTrainTools.snakeTrainTools.getAllMaxIndex(trainSnakeEvo.superviseAnswer(self.gameSize,game,result,network)))

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
                if(game.fruit != fruitSave):
                    moveSinceLastFood = 0
                    errors = [0]*4
                    errors[answerIndex] = 1-result[answerIndex]
                    network.backward(errors)
                else:
                    moveSinceLastFood += 1
                    currentDistance = abs(game.snake[-1][0]-game.fruit[0])+abs(game.snake[-1][1]-game.fruit[1])
                    previousDistance = abs(headSave[0]-game.fruit[0])+abs(headSave[1]-game.fruit[1])
                    if(currentDistance < previousDistance):
                        errors = [0]*4
                        errors[answerIndex] = (1-result[answerIndex])* 0.5
                        network.backward(errors)
                    elif(currentDistance > previousDistance):
                        errors = [0]*4
                        errors[answerIndex] = -result[answerIndex]*0.5
                        network.backward(errors)
                if(moveSinceLastFood >= self.gameSize**2):
                    state = False

            currentPerformance.append(len(game.snake))
        return sum(currentPerformance)/len(currentPerformance)
    
    def superviseAnswer(gameSize : int,game : snakeGame.Game,result : list[float],network : classe.Networks) -> int:
        gameGrid = game.getGrid()
        modifiedResult = result.copy()
        errors = [i for i in result]*4
        try:
            if(game.snake[-1][1]-1 < 0 or gameGrid[game.snake[-1][1]-1,game.snake[-1][0]] == -1):
                errors[0] = -result[0]
                modifiedResult[0] = 0
            if(game.snake[-1][1]+1 >= gameSize or gameGrid[game.snake[-1][1]+1,game.snake[-1][0]] == -1):
                errors[1] = -result[1]
                modifiedResult[1] = 0
            if(game.snake[-1][0]-1 < 0 or gameGrid[game.snake[-1][1],game.snake[-1][0]-1] == -1):
                errors[2] = -result[2]
                modifiedResult[2] = 0
            if(game.snake[-1][0]+1 >= gameSize or gameGrid[game.snake[-1][1],game.snake[-1][0]+1] == -1):
                errors[3] = -result[3]
                modifiedResult[3] = 0
            return modifiedResult
        except IndexError:
            print(game.snake[-1])
    
    def generateInput(grid : list[list[int]],seeAllMap : bool,snake : list[list[int]]) -> list[int]:
        networkInput = []
        snakeHead = snake[-1]
        gameSize = len(grid)
        if(True):
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
