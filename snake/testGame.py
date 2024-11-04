import snakeGame
import threading

game = snakeGame.Game(5)

def runGame(result):
    ui.input_field.delete(0,)
    if(result == "1"):
        game.directionY = -1
        game.directionX = 0
    elif(result == "2"):
        game.directionY = 1
        game.directionX = 0
    elif(result == "3"):
        game.directionX = -1
        game.directionY = 0
    elif(result == "4"):
        game.directionX = 1
        game.directionY = 0
    else:
        raise KeyboardInterrupt("")
    
    game.update()
    ui.grid = game.getGrid()
    if(game.checkState() != None):exit(0)

ui = snakeGame.UI(5,userInput=True,inputHandler=runGame)
ui.startGame(game.getGrid())