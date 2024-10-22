import snakeGame

game = snakeGame.Game(7)

state = None

while(state == None):
    print(game.getGrid(),game.snake,game.fruit)
    result = input("Ou voulez bouger ?\n1 : Haut\n2 : Bas\n3 : Gauche\n4 : Droite\n")

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
    state = game.checkState()