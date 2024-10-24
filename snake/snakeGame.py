import numpy as np
import random
import tkinter as tk

class Game():
    def __init__(self,size : int) -> None:
        self.size = size
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
        new = [random.randint(0,self.size-1),random.randint(0,self.size-1)]
        while(new in self.snake):
            new = [random.randint(0,self.size-1),random.randint(0,self.size-1)]
        self.fruit = new

    def isFruitHere(self) -> bool:
        return self.fruit != None

    def update(self):
        self.moveHead()
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
        grid[self.fruit[1]][self.fruit[0]] = 1
        return grid

    def checkState(self) -> bool:
        if(self.snake[len(self.snake)-1][0] < 0 or self.snake[len(self.snake)-1][1] < 0):return False
        if(self.snake[len(self.snake)-1][0] >= self.size or self.snake[len(self.snake)-1][1] >= self.size):return False
        if(self.snake.count(self.snake[len(self.snake)-1]) > 1):return False
        if(len(self.snake) == self.size**2):return True
        return None

class UI:
    def __init__(self,size : int,grid : list[list[int]],userInput : bool,inputHandler) -> None:
        self.size = size
        self.grid = grid
        self.userInput = userInput
        self.inputHandler = inputHandler

    def handleInput(self,event=None):
        self.inputHandler(self.input_field.get())
        self.input_field.delete(0, tk.END)

    def start(self):
        self.root = tk.Tk()
        self.root.title("Snake Game")
        self.canvas = tk.Canvas(self.root, width=self.size * 20, height=self.size * 20)
        self.canvas.pack()
        self.input_field = tk.Entry(self.root)
        self.input_field.pack()
        self.input_field.bind("<Return>",self.handleInput)
        self.update_game()
        self.root.mainloop()

    # Mise à jour du jeu, à l'avenir, on peut ajouter plus de logique pour animer le serpent
    def update_game(self):
        self.draw_game()
        self.root.after(100, self.update_game)

    def draw_game(self):
        self.canvas.delete("all")
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == -1:  # Corps du serpent
                    self.canvas.create_rectangle(j * 20, i * 20, (j + 1) * 20, (i + 1) * 20, fill="green")
                elif self.grid[i][j] == 1:  # Nourriture
                    self.canvas.create_oval(j * 20, i * 20, (j + 1) * 20, (i + 1) * 20, fill="red")
