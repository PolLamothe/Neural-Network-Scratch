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

ERROR_REVIEW_SIZE = 6

WINNED_GAME_REVIEW_SIZE = 5

PACKED_BODY_COEFF = 0.1

WINNED_GAME_SIZE = 100

class trainSnakeEvo():
    def __init__(self, gameSize: int, averageAim: int, network : classe.CNN) -> None:
        self.gameSize = gameSize
        self.averageAim = averageAim
        self.network = network
    
    def train(self) -> classe.NN:
        startingTime = time.time()
        winnedGames = []
        correctedSituations = {}

        lastPerformance = [0 for i in range(MEAN_SIZE)]
        count = 0
        maxAverage = 0
        winnedGameReviewCount = 0
        winnedGameMostSeen = 0
        shortestWinnedGame = -1
        explorationCount = 0

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
                if(explorationCount >= WINNED_GAME_REVIEW_SIZE):
                    explorationCount = 0
                    answerIndex = [i for i in range(4)]
                    answerProb = []
                    for value in supervisedResult:
                        if(value == -1):
                            answerProb.append(0)
                        elif(value == -0.5):
                            if(min(supervisedResult) == -0.5):
                                answerProb.append(1)
                            else:
                                answerProb.append(0)
                        else:
                            answerProb.append((value*5)**3)
                    

                    if(max(answerProb) == 0):
                        answerProb = [1 for i in range(4)]

                    answerIndex = random.choices(answerIndex,weights=answerProb)[0]
                else:
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
                snakeSave = game.snake.copy()
                game.update()
                state = game.checkState()

                self.network.backward(np.array([self.getError(
                    snakeSave,fruitSave,game.snake,game.fruit,result,supervisedResult,answerIndex,True
                )]))

                previousData.append({
                    "snake":copy.deepcopy(game.snake),
                    "index":answerIndex,
                    "fruit":copy.deepcopy(game.fruit),
                    "original" : True,
                    "forbidden" : None,
                    "result" : copy.deepcopy(supervisedResult)
                })
                if(game.fruit != fruitSave):
                    moveSinceLastFruit = 0
                else:
                    moveSinceLastFruit += 1
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

                        rotatedGames = trainSnakeEvo.rotateGame(filtred[-1]["snake"],filtred[-1]["fruit"],self.gameSize)
                        previousRotatedGames = trainSnakeEvo.rotateGame(filtred[-2]["snake"],filtred[-2]["fruit"],self.gameSize)
                        for index,rotatedGame in enumerate(rotatedGames):
                            tempGame = snakeGame.Game(self.gameSize)
                            tempGame.snake = copy.deepcopy(rotatedGame[0])
                            tempGame.fruit = copy.deepcopy(rotatedGame[1])

                            Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                            result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                            supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,tempGame.snake,result,[])
                            answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(supervisedResult))

                            error = [0 for i in range(4)]
                            for i in range(4):
                                if(supervisedResult[i] == -1):
                                    error[i] = -result[i]

                            if(len(tempGame.snake)-4) in correctedSituations:
                                correctedSituations[len(tempGame.snake)-4].append({
                                    "rotation" : index,
                                    "rotatedGame" : copy.deepcopy(rotatedGame),
                                    "previousRotatedGame" : copy.deepcopy(previousRotatedGames[index]),
                                    "bannedIndex" : self.get_aligned_answer(index,bannedIndex),
                                    "recommandedIndex" : self.get_aligned_answer(index,recommandedIndex),
                                    })
                            else:
                                correctedSituations[len(tempGame.snake)-4] = [{
                                    "rotation" : index,
                                    "rotatedGame" : copy.deepcopy(rotatedGame),
                                    "previousRotatedGame" : copy.deepcopy(previousRotatedGames[index]),
                                    "bannedIndex" : self.get_aligned_answer(index,bannedIndex),
                                    "recommandedIndex" : self.get_aligned_answer(index,recommandedIndex),
                                    }]
                            
                            #error[self.get_aligned_answer(index,recommandedIndex)] = (1-result[self.get_aligned_answer(index,recommandedIndex)])
                            error[self.get_aligned_answer(index,bannedIndex)] = -result[self.get_aligned_answer(index,bannedIndex)]
                            self.network.backward(np.array([error]))
                if(state == True):
                    moveSinceLastFruit = 0
                    if(shortestWinnedGame == -1 or shortestWinnedGame > len(previousData)):
                        shortestWinnedGame = len(previousData)
                    winnedGames.append({"seen":len(winnedGames),"data":copy.deepcopy(previousData)})
            count += 1

            if(explorationCount < WINNED_GAME_REVIEW_SIZE):
                lastPerformance.append(len(game.snake))
                lastPerformance.pop(0)

            explorationCount += 1

            if(len(correctedSituations) > ERROR_REVIEW_SIZE):
                lengthToReview = []
                for i in range(min(lastPerformance),max(lastPerformance)):
                    if(i != self.gameSize**2 and i in list(correctedSituations.keys())):
                        lengthToReview.append(i)
                correctedSituationsSelected = [random.choice(correctedSituations[random.choice(lengthToReview)]) for i in range(ERROR_REVIEW_SIZE)]

                for correctedSituation in correctedSituationsSelected:
                    tempGame = snakeGame.Game(self.gameSize)
                    tempGame.snake = copy.deepcopy(correctedSituation["rotatedGame"][0])
                    tempGame.fruit = copy.deepcopy(correctedSituation["rotatedGame"][1])

                    Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                    result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                    supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,tempGame.snake,result,[])
                    answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(supervisedResult))
                    
                    error = [0 for i in range(4)]
                    for i in range(4):
                        if(supervisedResult[i] == -1):
                            error[i] = -result[i]

                    error[correctedSituation["bannedIndex"]] = -result[correctedSituation["bannedIndex"]]
                    #error[correctedSituation["recommandedIndex"]] = 1-result[correctedSituation["recommandedIndex"]]
                    self.network.backward(np.array([error]))

            if(winnedGameReviewCount >= WINNED_GAME_REVIEW_SIZE and len(winnedGames) > 0):
                winnedGameReviewCount = 0

                winnedGamesProb = []

                for game in winnedGames:
                    if(game["seen"] > winnedGameMostSeen):
                        print("ERREUR DAOBOZABFA : ",winnedGameMostSeen,game["seen"])
                    winnedGamesProb.append(winnedGameMostSeen-game["seen"]+1)

                selectedGameIndex = random.choices(range(len(winnedGames)),weights=winnedGamesProb,k=1)[0]
                selectedGame = winnedGames[selectedGameIndex]["data"]
                winnedGames[selectedGameIndex]["seen"] += len(winnedGames)
                if winnedGames[selectedGameIndex]["seen"] > winnedGameMostSeen:
                    winnedGameMostSeen = winnedGames[selectedGameIndex]["seen"]

                indexAboveMean = None
                for (index,data) in enumerate(selectedGame):
                    if(len(data["snake"]) > min(lastPerformance)):
                        indexAboveMean = index
                        break

                rotatedGames = [[],[],[],[]]

                for i in range(indexAboveMean,len(selectedGame)-1):
                    rotatedGame = trainSnakeEvo.rotateGame(selectedGame[i]["snake"],selectedGame[i]["fruit"],self.gameSize)

                    for index,rotatedGame in enumerate(rotatedGame):
                        rotatedIndex = self.get_aligned_answer(index,selectedGame[i+1]["index"])

                        rotatedGames[index].append(rotatedGame+[rotatedIndex])

                for x in range(4):

                    for i in range(len(rotatedGames[x])-1):                       
                        tempGame = snakeGame.Game(self.gameSize)
                        tempGame.snake = copy.deepcopy(rotatedGames[x][i][0])
                        tempGame.fruit = copy.deepcopy(rotatedGames[x][i][1])

                        Networkinput = trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake)
                        result = self.network.forward(np.array([np.array(Networkinput)]))[0]
                        supervisedResult = trainSnakeEvo.superviseAnswer(self.gameSize,tempGame.snake,result.tolist(),[])
                        answerIndex = random.choice(trainSnakeEvo.getAllMaxIndex(supervisedResult))

                        error = self.getError(
                            rotatedGames[x][i][0],
                            rotatedGames[x][i][1],
                            rotatedGames[x][i+1][0],
                            rotatedGames[x][i+1][1],
                            result,
                            supervisedResult,
                            answerIndex,
                            False
                        )

                        if(error[rotatedGames[x][i][2]] == 0):
                            error[rotatedGames[x][i][2]] = (1-result[rotatedGames[x][i][2]]) * (shortestWinnedGame/len(selectedGame))

                        self.network.backward(np.array([error]))
            else:
                winnedGameReviewCount += 1
        print("\nThe training is over !")
        return copy.deepcopy(self.network)
    
    def benchmarkModel(network : classe.NN,gameSize : int) -> float:
        lengthHistory = []
        for i in range(500):
            if(len(lengthHistory) > 0):
                print("benchmark at ",i/5,"%, current mean : ",round(sum(lengthHistory)/len(lengthHistory),2),end="\r")
            state = None
            game = snakeGame.Game(gameSize)
            moveSinceLastFruit = 0
            previousData = []
            while(state == None):
                Networkinput = trainSnakeEvo.generateInput(game.getGrid(),game.snake)
                result = network.forward(np.array([Networkinput]))[0]
                supervisedResult = trainSnakeEvo.superviseAnswer(gameSize,game.snake,result,copy.deepcopy(previousData))
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
                snakeSave = copy.deepcopy(game.snake)
                game.update()
                state = game.checkState()
                previousData.append({
                    "snake":copy.deepcopy(snakeSave),
                    "index":answerIndex,
                    "fruit":copy.deepcopy(game.fruit),
                    "original" : True,
                    "forbidden" : None,
                    "result" : copy.deepcopy(supervisedResult)
                    })

                if(len(game.snake) >= gameSize**2):
                    state = True
                
                if(game.fruit != fruitSave):
                    moveSinceLastFruit = 0
                else:
                    moveSinceLastFruit += 1
                
                if(moveSinceLastFruit > gameSize**2*3):
                    state = False

            lengthHistory.append(len(copy.deepcopy(game.snake)))
        print("The agent achieved an average length of ",round(sum(lengthHistory)/len(lengthHistory),2)," on ",500," games")
        return sum(lengthHistory)/len(lengthHistory)
    
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
                previousDataCopy.append({
                    "snake":gameCopy.snake,
                    "index":i,
                    "fruit":gameCopy.fruit,
                    "forbidden" : None,
                    "original" : False,
                    "result" : copy.deepcopy(supervisedResult),
                    })
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
        
    def getError(self,
                previousSnake : list[list[int]],
                previousFruit : list[int],
                currentSnake : list[list[int]],
                currentFruit : list[int],
                result : list[float],
                supervisedResult : list[float],
                answerIndex : int,
                distanceReward : bool) -> np.ndarray:
        
        error : list[float]= [0]*4
        for i in range(4):
            if(supervisedResult[i] == -1):
                error[i] = -result[i]

        if(currentFruit != previousFruit):
            error[answerIndex] = 1-result[answerIndex]
        elif distanceReward:
            currentDistance = abs(currentSnake[-1][0]-currentFruit[0])+abs(currentSnake[-1][1]-currentFruit[1])
            previousDistance = abs(previousSnake[-1][0]-currentFruit[0])+abs(previousSnake[-1][1]-currentFruit[1])
            if(currentDistance < previousDistance):
                if(error[answerIndex] == 0):
                    error[answerIndex] = (1-result[answerIndex]) * ((math.log(1)-math.log(len(currentSnake)/(self.gameSize**2)))/2)
                #errors[answerIndex] = (1-result[answerIndex]) * max(0.5-(len(game.snake)/(self.gameSize**2)),0)
            elif(currentDistance > previousDistance):
                if error[answerIndex] == 0:
                    error[answerIndex] = -result[answerIndex] * ((math.log(1)-math.log(len(currentSnake)/(self.gameSize**2)))/2)

        previousPackedBody = getPackedBody(previousSnake)
        
        packedBody = getPackedBody(currentSnake)
        if(packedBody < previousPackedBody):
            if(error[answerIndex] < 0):
                error[answerIndex] *= 1 + PACKED_BODY_COEFF
            if(error[answerIndex] > 0):
                error[answerIndex] *= 1 - PACKED_BODY_COEFF
        elif(packedBody > previousPackedBody):
            if(error[answerIndex] < 0):
                error[answerIndex] *= 1 - PACKED_BODY_COEFF
            if(error[answerIndex] > 0):
                error[answerIndex] *= 1 + PACKED_BODY_COEFF
                    
        zonesNumberSave = getSeparatedZones(previousSnake,self.gameSize)
        
        zonesNumber = getSeparatedZones(currentSnake,self.gameSize)
        if(zonesNumberSave < zonesNumber):
            error[answerIndex] = -result[answerIndex]
        elif(zonesNumberSave > zonesNumber):
            error[answerIndex] = 1-result[answerIndex]
        
        return error
    
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

    def rotate_point(x: int, y: int, alignment: int,gameSize : int) -> list[int, int]:
        if alignment == 0:
            return [x, y]
        elif alignment == 1:  # 90°
            return [y, gameSize - 1 - x]
        elif alignment == 2:  # 180°
            return [gameSize - 1 - x, gameSize - 1 - y]
        elif alignment == 3:  # 270°
            return [gameSize - 1 - y, x]
    
    def rotate_agent_result(result : list[float],alignment : int) -> list[float]:
        if(alignment == 0):
            return result
        elif(alignment == 1):
            return [result[2],result[3],result[1],result[0]]
        elif(alignment == 2):
            return [result[1],result[0],result[3],result[2]]
        elif(alignment == 3):
            return [result[3],result[2],result[0],result[1]]

    def rotateGame(snake : list[list[int]],fruit : list[int],gameSize : int) -> list[list[list[int]]]:
        result = []

        for j in range(4):

            alignedSnake = copy.deepcopy(snake)

            for i in range(len(alignedSnake)):
                alignedSnake[i] = trainSnakeEvo.rotate_point(alignedSnake[i][0],alignedSnake[i][1],j,gameSize)
            
            alignedFruit = trainSnakeEvo.rotate_point(fruit[0],fruit[1],j,gameSize)
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
        
        for i in range(len(previousData)-2,0,-1):
            if(previousData[i]["snake"] == snake):
                currentResultData = sort_with_index(previousData[i]["result"])
                diff = abs(currentResultData[0][0]-currentResultData[1][0])
                smallerFound = False
                for j in range(i+1,len(previousData)-2):
                    currentResultData1 = sort_with_index(previousData[j]["result"])
                    diff1 = abs(currentResultData1[0][0]-currentResultData1[1][0])
                    if diff1 < diff:
                        smallerFound = True
                        break
                if(not smallerFound):
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

def sort_with_index(lst : list[float]) -> list[list]:
    return [[v, i] for i, v in sorted(enumerate(lst), key=lambda x: x[1])]

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

def getPackedBody(snake : list[list[int]]) -> int:
    nb_collisions = 0
    n = len(snake)

    for i in range(n):
        x1, y1 = snake[i]
        for j in range(i + 1, n):
            # on ignore les voisins directs (consécutifs dans la liste)
            if abs(i - j) == 1:
                continue

            x2, y2 = snake[j]

            # test si les deux cases sont collées (adjacentes 4-connexité)
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                nb_collisions += 1

    return nb_collisions

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
        print(filenames)
        for filename in filenames:
            if(filename.split(".")[1] == "pkl"):
                allModel.append("".join(filename))

    bestModel = None

    print(allModel)

    with open("../snake/model/trainedData.json","r") as file:
        jsonData = json.loads(file.read())
        for i in range(len(allModel)):
            if(bestModel == None or jsonData[bestModel]["aim"] < jsonData[allModel[i].split("_")[1]]["aim"]):
                bestModel = allModel[i].split("_")[1]
    
    gameSize = jsonData[bestModel]["gameSize"]

    with open("../snake/model/snake_"+bestModel+"_.pkl", "rb") as file:
        network : classe.NN = pickle.load(file)
    
    game = snakeGame.Game(gameSize)
    moveSinceLastFood = 0
    previousData = []
    data = []

    data.append({
        "fruit":copy.deepcopy(game.fruit),
        "snake":copy.deepcopy(game.snake)
    })

    while(game.checkState() == None):
        Networkinput = []
        rotatedGames = trainSnakeEvo.rotateGame(game.snake,game.fruit,game.size)
        for games in rotatedGames:
            tempGame = snakeGame.Game(gameSize)
            tempGame.snake = games[0]
            tempGame.fruit = games[1]
            Networkinput.append(trainSnakeEvo.generateInput(tempGame.getGrid(),tempGame.snake))
        result = network.forward(np.array(Networkinput))
        averageResult = np.mean([trainSnakeEvo.rotate_agent_result(_result,index) for index,_result in enumerate(result)],axis=0)
        supervisedResult = trainSnakeEvo.superviseAnswer(gameSize,game.snake,averageResult.tolist(),previousData)
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
        data.append({
            "fruit":copy.deepcopy(game.fruit),
            "snake":copy.deepcopy(game.snake)
        })
        previousData.append({
            "snake":copy.deepcopy(game.snake),
            "fruit":copy.deepcopy(game.fruit),
            "index":answerIndex,
            "result" : supervisedResult
            })
        if(fruitSave != game.fruit):
            moveSinceLastFood = 0

        moveSinceLastFood += 1
        
        '''if moveSinceLastFood > gameSize**2*3:
            break'''
    return data