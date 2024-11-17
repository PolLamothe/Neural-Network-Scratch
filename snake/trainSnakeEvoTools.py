import snakeTrainTools
import snakeGame
import classe
import random
import operator
import matplotlib.pyplot as plt

SELECTIONSIZE = 41 #21 because when selecting the 21 best agent we get 210 childs so it's a 10% selection rate

class trainSnakeEvo(snakeTrainTools.snakeTrainTools):
    def __init__(self, gameSize: int, averageAim: int, hiddenLayers: list[int] = [], activationFunction: callable = None, neuroneActivation: list = None) -> None:
        self.gameSize = gameSize
        self.averageAim = averageAim
        self.childPerformance = []
        self.child = [classe.Networks([gameSize**2*3]+hiddenLayers+[4],activation=activationFunction,learningRate=0.1,neuroneActivation=neuroneActivation) for i in range(SELECTIONSIZE*10)]
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
            self.childPerformance = []
            for i in range(len(self.child)):#benchmarking the childs
                self.childPerformance.append({"number" : (currentGeneration-1)*SELECTIONSIZE*10+i,"performance":self.__runChild(self.child[i]),"network":self.child[i]})
            self.childPerformance.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor += self.childPerformance[:SELECTIONSIZE]
            survivor.sort(reverse=True,key=operator.itemgetter("performance"))
            survivor = survivor[:SELECTIONSIZE]
            if(not filter(lambda x : x["performance"] >= self.averageAim,survivor)):break
            tempNetworks = []
            for i in range(len(survivor)):#creating the next childs
                for x in range(i+1,len(survivor)):
                    tempNetworks.append(classe.Networks([self.gameSize**2*3]+self.hiddenLayers+[4],activation=self.activationFunction,learningRate=0.1,neuroneActivation=self.neuroneActivation,parents=[survivor[i]["network"],survivor[x]["network"]]))
            self.child = tempNetworks.copy()
            self.iterationData.append({"generation":currentGeneration,"maxLength":survivor[0]["performance"],"averageLength":trainSnakeEvo.getAveragePerformance(survivor)})
            plt.plot([i for i in range(len(self.iterationData))],[self.iterationData[i]["averageLength"] for i in range(len(self.iterationData))])
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
                Networkinput = snakeTrainTools.snakeTrainTools.generateInput(game.getGrid(),True,game.snake)
                result = network.forward(Networkinput)
                answerIndex = random.choice(snakeTrainTools.snakeTrainTools.getAllMaxIndex(self.__superviseAnswer(game,result,network)))

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
                if(game.fruit != fruitSave):
                    moveSinceLastFood = 0
                    errors = [0]*4
                    errors[answerIndex] = 1-result[answerIndex]
                    network.backward(errors)
                else:
                    moveSinceLastFood += 1
                if(moveSinceLastFood >= self.gameSize**2):
                    state = False

            currentPerformance.append(len(game.snake))
        return sum(currentPerformance)/len(currentPerformance)
    
    def __superviseAnswer(self,game : snakeGame.Game,result : list[float],network : classe.Networks) -> int:
        gameGrid = game.getGrid()
        modifiedResult = result.copy()
        errors = [0]*4
        try:
            if(game.snake[-1][1]-1 < 0 or gameGrid[game.snake[-1][1]-1,game.snake[-1][0]] == -1):
                errors[0] = -result[0]
                modifiedResult[0] = 0
            if(game.snake[-1][1]+1 >= self.gameSize or gameGrid[game.snake[-1][1]+1,game.snake[-1][0]] == -1):
                errors[1] = -result[1]
                modifiedResult[1] = 0
            if(game.snake[-1][0]-1 < 0 or gameGrid[game.snake[-1][1],game.snake[-1][0]-1] == -1):
                errors[2] = -result[2]
                modifiedResult[2] = 0
            if(game.snake[-1][0]+1 >= self.gameSize or gameGrid[game.snake[-1][1],game.snake[-1][0]+1] == -1):
                errors[3] = -result[3]
                modifiedResult[3] = 0
            #network.backward(errors)
            return modifiedResult
        except IndexError:
            print(game.snake[-1])