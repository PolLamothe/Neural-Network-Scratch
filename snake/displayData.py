import snakeGame
import json
from os import walk

allData = []
for (dirpath, dirnames, filenames) in walk("./data"):
    allData.extend(filenames)

grid = None

gameIndex = 0
gridIndex = 0

iterationAvaible = []

def selectionHandler(name : str):

    with open("./data/"+name, 'r') as fichier:
        jsonData = json.load(fichier)

    for i in range(len(jsonData["data"])):
        iterationAvaible.append(jsonData["data"][i]["iteration"])

    def updateData():
        global gridIndex
        gridIndex += 1
        try:
            ui.grid = jsonData["data"][gameIndex]["data"][gridIndex]
        except IndexError:
            return "GameOver"
        
    def replayData():
        global gridIndex
        gridIndex = 0
        ui.grid = jsonData["data"][gameIndex]["data"][gridIndex]

    def iterationChoosed(iteration : int):
        global ui
        global gameIndex
        for i in range(len(jsonData["data"])):
            if(jsonData["data"][i]["iteration"] == iteration):
                gameIndex = i
        ui.startGame(jsonData["data"][gameIndex]["data"][gridIndex],updateGrid=updateData,replayGame=replayData)
    
    ui.startChoosingMenu(iterationAvaible,iterationChoosed)

ui = snakeGame.UI()
ui.startChoosingDataMenu(allData,selectionHandler)