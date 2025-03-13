# Multilayer Perceptron from scratch

# Presentation

![demo](./assets/demo.png)

This project aim to help me understanding how IA works. That's why I choosed to use only numpy and math.

# List of the projects : 
 - An `multilayer Perceptron` and a `CNN` capable of detecting the number drawn in a 28 x 28 square (data provided by [MNIST](https://en.wikipedia.org/wiki/MNIST_database)).

 - An `multilayer Perceptron` capable of doing a good score in the snake game

 - An `CNN` capable of classifying pictures of planes,ships or cars
---
# File

- **classe.py** (This file contain all of the class and function used to create IA)

# Avaibles architectures :

## Simple Neural Network :

### Layers included in this architecture : 

- **Fully Connected Layer** : This layer is responsible of predicting the expected ouput based on it's previous errors.

- **Dropout Layer** : This layer is responsible of randomly disabling some neurones of the previous layer to prevent the network from overfiting to a specific pattern or the training data (This layer must be desactivate after training).

![Illustration](./assets/FCL.webp)

## Convolutional Neural Network :

### Layers included in this architecture :

- **Convolutional Layer** : This layer is reponsible of extracting features in an image.

- **Batch Normalization Layer** : This layer is responsible of making the result of the **Convolutional Layers** more "smooth".

- **Max Pooling Layer** : This layer is responsible of reducing the size of the result from the **Convolutional Layers**.

- **All Simple Neural Network's Layers** : This part is responsible of treating the data extracted by the convolutional part.

![Illustration](./assets/CNN.jpg)
---
# Math

## Fully Connected Layers :

### Variables :
- $X$ is the input of the layer
- $W$ is the weight of the layer
- $B$ is the bias of the layer
- $f$ is the activation function
- $f'$ is the derivative of the activation function
- $l$ is the learning rate
- $Y$ is the output of the layer
- $dY$ is the output error
- $dW$ is weight grandiant of the layer
- $db$ is bias gradiant of the layer
- $dX$ is the input error of the layer

### Forward Propagation :
$$Y = f(X \cdot W+B)$$

### Backward Propagation : 

#### Error adjustment
$$dY' = dY * f'(Y)$$

#### Weight gradiant :
$$dW = dY' \cdot X * l$$

#### Bias gradiant : 
$$dB = dY * l$$

#### Input error :
$$dX = dY \cdot W$$

## Convolutional Layers :

### Variables :

- $K$ is the kernel

### Forward Propagation :

$$ Yi = Bi + \sum_{j=1}^{n} Xj \bigstar Kij, i = 1..d$$

### Backward Propagation :

#### Kernel gradiant :
$$ dK = X \bigstar dY $$

#### Bias gradiant :
$$ dB = dY $$

#### Input error :
$$dXj = \sum_{i=1}^{n} dYi \underset{full}{*} Kij $$

## Batch Normalization :

### Variables :

- $\gamma$ is the scale of the layer

- $\beta$ is the gap of the layer

- $\epsilon$ is epsilon (it's a really small number used to avoid division by 0)

- $m$ is the number of sample in each batch


### Forward Propagation :

#### Mean :
$$ \mu = \frac{1}{m} \sum_{i=1}^{m}x_i$$

#### Variance :
$$ \sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu)^2$$

**The mean $\mu$ and the variance $\sigma^2$ are distinct for each sample and common for every batch**

#### Normalized value :

$$ \^x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

#### Value after applying scale and gap :

$$ y_i = \gamma \^x_i + \beta $$

### Backward Propagation :

#### Scale gradiant :

$$ d\gamma = \sum_{i=1}^{m}d y_i \cdot \^x_i$$

#### Gap gradiant :

$$ d\beta = \sum_{i=1}^{m}d y_i $$

#### Input gradiant

$$ dx_i = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} * (dy - (\frac{1}{m} \sum_{i=1}^{m}dy[i]) * (\frac{1}{m} \sum_{i=1}^{m}dy[i] * \^x_i) )$$

---
# Sources :

## Perceptron :

*Math behind a Perceptron* : https://www.youtube.com/watch?v=kft1AJ9WVDk&t=551s

*Math behind a multilayer Perceptron* : https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

*List and explanation of activations functions* : https://medium.com/@sanjay_dutta/multi-layer-perceptron-and-backpropagation-a-deep-dive-8438cc8bcae6

*Visualization of a multilayer Perceptron solving MNIST* : https://www.youtube.com/watch?v=9RN2Wr8xvro

## CNN :

*What's a CNN* : https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07

*What's a convolution* : https://www.youtube.com/watch?v=KuXjwB4LzSA

*How does a convolutional layer work* : https://www.youtube.com/watch?v=Lakz2MoHy6o

*Visualization of a CNN solving MNIST* : https://www.youtube.com/watch?v=pj9-rr1wDhM

## Batch Normalization :

*What is Batch Normalization and math for forward propagation* : https://medium.com/towards-data-science/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739