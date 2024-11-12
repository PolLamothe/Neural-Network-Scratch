# Multilayer Perceptron from scratch

# Presentation

![demo](./demo.png)

This project aim to help me understanding how IA works. That's why I choosed to use only numpy and math.

# List of the projects : 
 - An IA capable of detecting the number draw in a 28 x 28 square (data provided by [MNIST](https://en.wikipedia.org/wiki/MNIST_database)).

 - A IA capable of doing a very good score in the snake game

---
# File

Here is the list of the file in this project and what are they used for :

- **classe.py** (This file contain all of the class and function used for the IA specifically)

- **uniquePerceptron.py** (This file contain a program to train and test the IA using only 1 neurones for a linear problem)

- **layerTest.py** (This file contain a program to train and test the IA using more than 1 layer of neurones for a more complex (XOR) problem)

### Number Detection :
- **numberDetection.py** : This file contain the code to train and test the IA in number detection. **use -c command to show help**

- **numberDetectionTools.py** : This file contain function used in **numberDetection.py** and **UI.py**

- **UI.py** : Use this file to see the number wich are given to the IA

### Snake :

 - **displayData.py** : A script to see the preregistered game played by the IA (include a UI)

 - **liveDemo.py** : A script to see the IA play directly (include a UI)

 - **testGame.py** : A script to test the snake game by playing it yourself (include a UI)

 - **snakeTrain.py** : The script that allow you to choose the parameter of the IA

 - **snakeTrainTools.py** : The script that train the IA

 - **snakeGame.py** : The script that contain the code for the snake game and the different UI

 - **data (folder)** : This folder contain all of the preregistered game played by IA

 - **model (folder)** : This folder contain all of the trained IA

---
# Math

## Variables :
- $X$ is the input of the nerone
- $W$ is the weight of the nerones connection
- $b$ is the bias of the nerone
- $f$ is the activation function
- $l$ is the learning rate
- $y$ is the output of the nerone
- $\sigma y$ is the output error
- $\sigma W$ is the new weight matrix for the nerone
- $\sigma b$ is the new bias of the nerone
- $\sigma X$ is the input error of the nerone `warning: it's not directly the output error of the previous layer`

## Forward Propagation :
$$y = f(X*W+b)$$

## Backward Propagation : 
### Weight adjustment :
$$\sigma W = W + \sigma y * f'(y) * X * l$$

### Bias adjustment : 
$$\sigma b = b + \sigma y * f'(y) * l$$

### Input error :
$$\sigma X = \sigma y * f'(y) * W * X$$

---
# Sources :

https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

https://medium.com/@sanjay_dutta/multi-layer-perceptron-and-backpropagation-a-deep-dive-8438cc8bcae6

https://www.youtube.com/watch?v=9RN2Wr8xvro

https://www.youtube.com/watch?v=kft1AJ9WVDk&t=551s