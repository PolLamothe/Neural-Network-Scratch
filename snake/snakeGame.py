import numpy as np
import random

class Game():
    def __init__(self,size : int) -> None:
        self.size = size
        self.directionX = 1
        self.directionY = 0
        self.snake = [[round(size/2)-1,round(size/2)-1]]
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
            if(case[1] > 0 and case[0] > 0 and case[1] < self.size and case[0] < self.size):
                grid[case[1]][case[0]] = -1
        grid[self.fruit[1]][self.fruit[0]] = 1
        return grid

    def checkState(self) -> bool:
        if(self.snake[len(self.snake)-1][0] < 0 or self.snake[len(self.snake)-1][1] < 0):return False
        if(self.snake[len(self.snake)-1][0] >= self.size or self.snake[len(self.snake)-1][1] >= self.size):return False
        if(len(self.snake) == self.size**2):return True
        return None