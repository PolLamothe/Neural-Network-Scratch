import numpy as np
import random
import tkinter as tk

class Game():
    def __init__(self,size : int) -> None:
        self.size = size
        if(size < 5):
            raise Exception("Map size must be at least 5")
        self.directionX = 1
        self.directionY = 0
        self.snake = []
        for i in range(3,-1,-1):
            self.snake.append([round(size/2)-1-i+2,round(size/2)])
        self.fruit = []
        self.newFruit()

    def moveHead(self):
        for i in range(-1,2,2):
            if(self.directionX == i):
                self.snake.append(self.snake[len(self.snake)-1].copy())
                self.snake[len(self.snake)-1][0] += i
        for i in range(-1,2,2):
            if(self.directionY == i):
                    self.snake.append(self.snake[len(self.snake)-1].copy())
                    self.snake[len(self.snake)-1][1] += i
        if(self.fruit in self.snake):
            self.fruit = None
    
    def removeTail(self):
        self.snake.pop(0)

    def newFruit(self):
        avaible = []
        for i in range(self.size):
            for x in range(self.size):
                if([i,x] not in self.snake):
                    avaible.append([i,x])
        self.fruit = random.choice(avaible)

    def isFruitHere(self) -> bool:
        return self.fruit != None

    def update(self):
        self.moveHead()
        if(self.checkState() == True):return
        if(self.snake[len(self.snake)-1][0] >= self.size or self.snake[len(self.snake)-1][1] >= self.size):return
        if(self.snake[len(self.snake)-1][0] < 0 or self.snake[len(self.snake)-1][1] < 0):return
        if(self.isFruitHere()):
            self.removeTail()
        else:
            self.newFruit()

    def getGrid(self) -> list[list[int]]:
        grid = np.zeros((self.size,self.size))
        for case in self.snake:
            if(case[1] >= 0 and case[0] >= 0 and case[1] < self.size and case[0] < self.size):
                grid[case[1]][case[0]] = -1
        try:
            grid[self.fruit[1]][self.fruit[0]] = 1
        except:pass
        return grid

    def checkState(self) -> bool:
        def checkDoublons(array : list[list[int]]) -> bool:
            for position in array:
                if(array.count(position) > 1):return False
            return True
        if(self.snake[len(self.snake)-1][0] < 0 or self.snake[len(self.snake)-1][1] < 0):return False
        if(self.snake[len(self.snake)-1][0] >= self.size or self.snake[len(self.snake)-1][1] >= self.size):return False
        if(not checkDoublons(self.snake)):return False
        if(len(self.snake) == self.size**2):return True
        return None

class UI:
    def handleInput(self,event):
        self.inputHandler(self.input_field.get())
        self.input_field.delete(0, tk.END)

    def handleReplay(self):
        self.replayGame()
        self.gameOverText.pack_forget()
        self.replayButton.pack_forget()
        self.update_game()

    def handeIterationChoice(self,iteration : int):
        self.root.destroy()
        self.choiceHandler(iteration)

    def startChoosingMenu(self,iterationsAvaible : list[int],choiceHandler : callable):
        '''
        Parameters
        ----------
        userInput : bool
            This parameter is used to know if the window will be playing a pre-registered game or if the player will play directly
        inputHandler : function
            This function will be called when an input is entered into the window
        updateGrid : function
            This function will be called when the game need to update the grid
        replayGame : function
            This function will be called when the game need to replay the game
        '''
        self.root = tk.Tk()
        self.choiceHandler = choiceHandler
        self.root.title("Snake Game")
        label = tk.Label(self.root,text="Select the versions of the IA wich you want to see play")
        label.pack()

        canvas = tk.Canvas(self.root, width=400, height=300)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.config(yscrollcommand=scrollbar.set)
        
        frame = tk.Frame(canvas)
        
        buttons = []
        for iteration in iterationsAvaible:
            buttons.append(tk.Button(frame,text=str(iteration)+" iterations",command=lambda : self.handeIterationChoice(iteration)))
        for button in buttons:
            button.pack()
        
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.root.mainloop()

    def startChoosingDataMenu(self,allData : list[str],selectionHandler : callable):
        self.root = tk.Tk()
        self.root.title("Snake Game")

        label = tk.Label(self.root,text="Select the IA wich you want to see play")
        label.pack()

        canvas = tk.Canvas(self.root, width=400, height=300)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.config(yscrollcommand=scrollbar.set)

        frame = tk.Frame(canvas)

        for dataName in allData:
            tempButton = tk.Button(frame,text=dataName,command=lambda:selectionHandler(dataName))
            tempButton.pack()

        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.mainloop()

    def startChoosingModel(self,allModel : list[str],selectionHandler : callable):
        self.root = tk.Tk()
        self.root.title("Snake Game")

        label = tk.Label(self.root,text="Select the IA wich you want to see play")
        label.pack()

        canvas = tk.Canvas(self.root, width=400, height=300)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.config(yscrollcommand=scrollbar.set)

        frame = tk.Frame(canvas)

        for modelName in allModel:
            tempButton = tk.Button(frame,text=modelName,command=lambda:selectionHandler(modelName))
            tempButton.pack()

        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.mainloop()

    def changeFreeze(self):
        if(self.freezeState == False):self.pauseButton.config(text="Unfreeze")
        else:self.pauseButton.config(text="Freeze")
        self.freezeState = not self.freezeState
        if(self.freezeState == False):
            self.update_game()

    def startGame(self, grid : list[list[int]],userInput=False,inputHandler : callable = None,updateGrid : callable = None,replayGame : callable = None):
        self.userInput = userInput
        self.inputHandler = inputHandler
        self.updateGrid = updateGrid
        self.replayGame = replayGame
        self.freezeState = False
        self.size = len(grid)
        self.cellSize = (20/self.size)*20
        self.grid = grid

        self.root = tk.Tk()
        self.root.title("Snake Game")
        self.canvas = tk.Canvas(self.root, width=self.size * self.cellSize, height=self.size * self.cellSize)
        self.canvas.pack()
        if(self.userInput == True):
            self.input_field = tk.Entry(self.root)
            self.input_field.pack()
            self.input_field.bind("<Return>",self.handleInput)
        else:
            self.pauseButton = tk.Button(self.root,text="Freeze",command=self.changeFreeze)
            self.pauseButton.pack()
        self.update_game()
        self.root.mainloop()

    def update_game(self):
        if(self.freezeState):return
        if(self.userInput == False):
            if(self.updateGrid() == "GameOver"):
                self.gameOverText = tk.Label(self.root,text="Game Over")
                self.gameOverText.pack()
                self.replayButton = tk.Button(self.root,text="Replay",command=self.handleReplay)
                self.replayButton.pack()
                return
        self.draw_game()
        self.root.after(200, self.update_game)

    def draw_game(self):
        self.canvas.delete("all")
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == -1:
                    self.canvas.create_rectangle(j * self.cellSize, i * self.cellSize, (j + 1) * self.cellSize, (i + 1) * self.cellSize, fill="green")
                elif self.grid[i][j] == 1:
                    self.canvas.create_oval(j * self.cellSize, i * self.cellSize, (j + 1) * self.cellSize, (i + 1) * self.cellSize, fill="red")