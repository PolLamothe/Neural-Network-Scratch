import numpy as np
import random

class Game():
    def __init__(self,size : int) -> None:
        self.grid = np.zeros((size,size))
        self.grid[round(size/2)-1][round(size/2)-1] = -1
        self.headPosition = [round(size/2)-1,round(size/2)-1]
        self.directionX = 1
        self.directionY = 0
        self.newFruit()
        self.length = 1

    def moveHead(self):
        headIndex = self.headPosition
        if(self.directionX == 1):
            self.grid[headIndex[1]][headIndex[0]+1] = -1
            self.headPosition[0] += 1
        elif(self.directionX == -1):
            self.grid[headIndex[1]][headIndex[0]-1] = -1
            self.headPosition[0] -= 1
        elif(self.directionY == 1):
            self.grid[headIndex[1]+1][headIndex[0]] = -1
            self.headPosition[1] += 1
        elif(self.directionY == -1):
            self.grid[headIndex[1]-1][headIndex[0]] = -1
            self.headPosition[1] -= 1

    def findNextCase(self,position : tuple[int,int],history : list[list[int,int]]) -> tuple[int,int]:
        if(self.grid[position[1]][position[0]+1] == -1 and [position[0]+1,position[1]] not in history):return [position[0]+1,position[1]]
        if(self.grid[position[1]][position[0]-1] == -1 and [position[0]-1,position[1]] not in history):return [position[0]-1,position[1]]
        if(self.grid[position[1]+1][position[0]] == -1 and [position[0],position[1]+1] not in history):return [position[0],position[1]+1]
        if(self.grid[position[1]-1][position[0]] == -1 and [position[0],position[1]-1] not in history):return [position[0],position[1]-1]
        return None
    
    def removeTail(self):
        history = []
        currentIndex = self.findNextCase(self.headPosition,history)
        history.append(currentIndex)
        while(currentIndex != None):
            currentIndex = self.findNextCase(currentIndex,history)
            if(currentIndex != None):history.append(currentIndex)
        print(history)
        self.grid[history[0][1],history[0][0]] = 0

    def newFruit(self):
        new = (random.randint(0,len(self.grid)-1),random.randint(0,len(self.grid)-1))
        while(self.grid[new[0]][new[1]] != 0):
            new = (random.randint(0,len(self.grid)-1),random.randint(0,len(self.grid)-1))
        self.grid[new[0]][new[1]] = 1

    def isFruitHere(self) -> bool:
        for line in self.grid:
            for column in line:
                if(column == 1):return True
        return False

    def update(self):
        self.moveHead()
        if(self.headPosition[0] >= len(self.grid) or self.headPosition[1] >= len(self.grid)):return
        if(self.headPosition[0] < 0 or self.headPosition[1] < 0):return
        if(self.isFruitHere()):
            self.removeTail()
        else:
            self.newFruit()
            self.length += 1

    def checkState(self) -> bool:
        count = 0
        for line in self.grid:
            for column in line:
                if(column == -1):count+=1
        if(self.length == len(self.grid)**2):return True
        if(count == self.length):return None
        return False