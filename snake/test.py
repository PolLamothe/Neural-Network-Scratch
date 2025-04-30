import math
import trainSnakeEvoTools
import snakeGame
import unittest
import random
import copy

class TestSuperviseAnswer(unittest.TestCase):
    #result = [top,bottom,left,right]
    def test_protectionFromBorder(self):
        
        game = snakeGame.Game(5)
        game.snake = [[0,0],[1,0],[2,0],[3,0],[4,0]]
        game.fruit = [3,3]
        '''
        TAIL -1 -1 -1 HEAD
         0    0  0  0  0
         0    0  0  0  0
         0    0  0  1  0
         0    0  0  0  0
        '''
        result = [1,1,1,1]
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(5,game.snake,result,[]),[-1,1,-1,-1])
    
    def test_protectionFromBody(self):
        game = snakeGame.Game(5)
        game.snake = [[1,0],[0,0],[0,1],[0,2],[1,2],[1,1]]
        game.fruit = [3,3]
        '''
        -1 TAIL 0 0 0
        -1 HEAD 0 0 0
        -1   -1 0 0 0
         0    0 0 1 0
         0    0 0 0 0
        '''
        result = [1,1,1,1]
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(5,game.snake,result,[]),[1,-1,-1,1])
    
    def test_canMoveToTail(self):
        game = snakeGame.Game(5)
        game.snake = [[0,0],[1,0],[1,1],[0,1]]
        game.fruit = [3,3]
        '''
        TAIL   -1 0 0 0
        HEAD   -1 0 0 0
        0       0 0 0 0
        0       0 0 1 0
        0       0 0 0 0
        '''
        result = [1,1,1,1]
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(5,game.snake,result,[]),[1,1,-1,-1])

    def test_loop(self):
        game = snakeGame.Game(5)
        game.snake = [[0,0],[1,0],[1,1],[0,1]]
        '''
        TAIL   -1 0 0 0
        HEAD   -1 0 0 0
        0       0 0 0 0
        0       0 0 0 0
        0       0 0 0 1
        '''
        game.fruit = [4,4]
        result = [1,1,1,1]
        previous = [#we simulate that the current situation has ever happen and that we already tried to go at the top
            {"grid":[
                [-1,-1,0,0,0],
                [-1,-1,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,1]
            ],
            "index":0,
            "snake":[[0,0],[1,0],[1,1],[0,1]]
            }
        ]
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(5,game.snake,result,previous),[-0.5,1,-1,-1])

    def test_getMaxIndex(self):
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.getAllMaxIndex([1,0,0,1]),[0,3])
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.getAllMaxIndex([-1,0,0,-1]),[1,2])
        self.assertEqual(trainSnakeEvoTools.trainSnakeEvo.getAllMaxIndex([-1,-0.5,-1,-1]),[1])

    def disabled_test_random(self):
        for i in range(100):
            print(i,end="\r")
            game = snakeGame.Game(5)
            dataSinceLastFood = []
            while(game.checkState() == None):
                supervisedResult = trainSnakeEvoTools.trainSnakeEvo.superviseAnswer(game.size,game.snake,[1,0.9,0.8,0.7],copy.deepcopy(dataSinceLastFood))
                answerIndex = random.choice(trainSnakeEvoTools.trainSnakeEvo.getAllMaxIndex(supervisedResult))

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
                snakeSave = copy.deepcopy(game.snake)
                game.update()
                if(fruitSave != game.fruit):
                    dataSinceLastFood = []
                else:
                    dataSinceLastFood.append({"snake":copy.deepcopy(snakeSave),"index":answerIndex})
            tempGrid = copy.deepcopy(game.getGrid()).tolist()
            for i in range(5):
                tempGrid[i] = [-1]+tempGrid[i]+[-1]
            tempGrid.append([-1]*7)
            tempGrid = [[-1]*7]+tempGrid
            error = False

            game.snake.pop()
            for i in range(-1,1,2):
                if(tempGrid[game.snake[-1][1]+1+i][game.snake[-1][0]+1] != -1):
                    error = True
            for i in range(-1,1,2):
                if(tempGrid[game.snake[-1][1]+1][game.snake[-1][0]+i+1] != -1):
                    error = True

            if(False):
                print(game.getGrid())
                print(game.snake)
                print(supervisedResult,answerIndex)
                self.fail()

class TestExploreEveryPossibility(unittest.TestCase):
    def test_firstCase(self):
        game = snakeGame.Game(5)
        game.snake = [[0,2],[0,1],[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[1,1]]
        game.fruit = [4,4]
        previousData = [
            {"snake" : [[1,4],[0,4],[0,3],[0,2],[0,1],[0,0],[1,0],[2,0],[2,1]],"fruit" : game.fruit,"index" : 1,"forbidden" : None,"original" : True},
            {"snake" : [[0,4],[0,3],[0,2],[0,1],[0,0],[1,0],[2,0],[2,1],[2,2]],"fruit" : game.fruit,"index" : 1,"forbidden" : None,"original" : True},
            {"snake" : [[0,3],[0,2],[0,1],[0,0],[1,0],[2,0],[2,1],[2,2],[1,2]],"fruit" : game.fruit,"index" : 2,"forbidden" : None,"original" : True},
            {"snake" : [[0,2],[0,1],[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[1,1]],"fruit" : game.fruit,"index" : 0,"forbidden" : None,"original" : True}
        ]
        '''
        -1   -1   -1 0 0
        -1   -1   -1 0 0
        -1   HEAD -1 0 0
        TAIL  0    0 0 0
        0     0    0 0 1
        '''
        '''for data in trainSnakeEvoTools.trainSnakeEvo.exploreEveryPossibility(game,previousData,len(previousData),True):
            print(data)'''

    def test_seenCase(self):
        previousData = [
            {'snake': [[1, 3], [2, 3], [2, 4], [3, 4], [3, 3]], 'index': 3, 'fruit': [1, 0], 'original': True, 'forbidden': None},
            {'snake': [[2, 3], [2, 4], [3, 4], [3, 3], [4, 3]], 'index': 1, 'fruit': [1, 0], 'original': True, 'forbidden': None}, 
            {'snake': [[2, 4], [3, 4], [3, 3], [4, 3], [4, 4]], 'index': 3, 'fruit': [1, 0], 'original': True, 'forbidden': None}]

        game = snakeGame.Game(5)
        game.snake = previousData[-1]["snake"]
        game.fruit = previousData[-1]["fruit"]
        for data in trainSnakeEvoTools.trainSnakeEvo.exploreEveryPossibility(game,previousData,len(previousData),True):
            print(data)

if __name__ == "__main__":
    unittest.main()