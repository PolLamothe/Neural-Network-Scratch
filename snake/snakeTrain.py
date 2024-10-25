import sys
import snakeGame
import numpy as np
import threading
import json

sys.path.append("../")
import classe

with open('./data/game.json', 'w') as json_file:
    json.dump([],json_file)

gameSize = 5

network = classe.Networks([8*2,4],classe.sigmoid,0.1)

state = None

previousData = []
previousResult = []
previousGrid = []

gameLength = 0
maxlength = 4

game = snakeGame.Game(gameSize)

iteration = 0
previousIteration = 0

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def reset():
    global game
    global state
    global previousData
    global gameLength
    global iteration
    global previousResult
    global previousGrid
    iteration += 1
    game = snakeGame.Game(gameSize)
    state = None
    previousData = []
    previousResult = []
    previousGrid = []

def generateInput():
    networkInput = []
    grid = game.getGrid()
    snakeHead = game.snake[len(game.snake)-1]
    for j in range(2):
        for i in range(-1,2):
            for x in range(-1,2):
                if(i != 0 or x != 0):
                    if((snakeHead[1]+i >= 0 and snakeHead[0]+x >= 0 and snakeHead[1]+i < game.size and snakeHead[0]+x < game.size)):
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
def checkIA():
    while(True):
        input("")
        print(network.forward(generateInput()))

thread = threading.Thread(target=checkIA)
thread.start()

while(state != True):
    print("nombre d'itération :"+str(iteration)+" longueur max : "+str(maxlength),end="\r")

    input = generateInput()
    result = network.forward(input)
    previousResult.append(result)
    #print(result)
    previousData.append(input)
    
    answerIndex = np.argmax(previousResult[len(previousResult)-1])

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
    if(game.fruit != fruitSave):
        #print("il a mangé un fruit !")
        #for i in range(len(previousData)):
        errors = [0]*4
        errors[answerIndex] = 1
        network.backward(errors)
        previousData = []
        previousResult = []

    state = game.checkState()
    if(state == False or len(previousData) >= 20):
        #for i in range(len(previousData)):
        errors = [0]*4
        errors[answerIndex] = -1#/len(previousData)
        network.backward(errors)
        if(len(game.snake) > maxlength):
            maxlength = len(game.snake)
        
        if(iteration == previousIteration*10 or iteration == 0):
            with open('./data/game.json', 'r') as fichier:
                jsonData = json.load(fichier)
            with open('./data/game.json', 'w') as json_file:
                jsonData.append({"iteration":iteration,"data":previousGrid})
                json.dump(jsonData,json_file,cls=NumpyArrayEncoder,indent=4)
            if(iteration == 0):
                previousIteration = 1
            else:previousIteration = iteration

        reset()
    elif(state == True):
        with open('./data/game.json', 'r') as fichier:
                jsonData = json.load(fichier)
        with open('./data/game.json', 'w') as json_file:
            jsonData.append({"iteration":iteration,"data":previousGrid})
            json.dump(jsonData,json_file,cls=NumpyArrayEncoder,indent=4)
        print("gagné")
        break