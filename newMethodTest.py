import numpy as np
import math

def fonction(x):
	return x**2
    
def dfonction(x):
	return x
    
n = 10#output
m = 60#input
   
W = np.random.rand(n,m)

X = np.random.rand(m)

B = np.random.rand(n)

Y = fonction(np.dot(W,X)+B)

E = np.random.rand(n)

newW = (W + (E * dfonction(Y))[:,np.newaxis]*X)
state = True
for i in range(len(newW)):
    madeOneByOne = W[i] + E[i] * dfonction(Y[i]) * X
    for x in range(len(newW[i])):
        if(newW[i][x] != madeOneByOne[x]):
            state = False
if(state):
    print("it works !")
else:
    print("it don't works")