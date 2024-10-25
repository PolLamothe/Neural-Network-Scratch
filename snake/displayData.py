import snakeGame
import json

with open('./data/game.json', 'r') as fichier:
    jsonData = json.load(fichier)

grid = None

gameIndex = 0
gridIndex = 0

for i in range(len(jsonData)):
    if(jsonData[i]["iteration"] == 672):
        gameIndex = i


def updateData():
    global gridIndex
    gridIndex += 1
    try:
        ui.grid = jsonData[gameIndex]["data"][gridIndex]
    except IndexError:
        return "GameOver"
    
def replayData():
    global gridIndex
    gridIndex = 0
    ui.grid = jsonData[gameIndex]["data"][gridIndex]

ui = snakeGame.UI(5,jsonData[gameIndex]["data"][gridIndex],False,updateGrid=updateData,replayGame=replayData)
ui.start()