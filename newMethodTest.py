import numpy as np
import math

def fonction(x):
	return x**2
    
def dfonction(x):
	return x
    
n = 2#output
m = 3#input
   
W = np.random.rand(n,m)

X = np.random.rand(m)

B = np.random.rand(n)

Y = fonction(np.dot(W,X)+B)

E = np.random.rand(n)

print("verifying the new Weight")
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

print("verifying the new Bias")
newB = (B+E*dfonction(Y))
state = True
for i in range(len(newB)):
    madeOneByOne = B[i] + E[i] * dfonction(Y[i])
    if(newB[i] != madeOneByOne):
        state = False
if(state):
    print("it works !")
else:
    print("it don't works")

print("verifying the new Input error")
newError = (np.dot(E*Y,W*X))
state = True

madeOneByOne = []
for i in range(len(newB)):
    madeOneByOne.append(E[i] * dfonction(Y[i]) * W[i] * X)
madeOneByOne = np.array(madeOneByOne).sum(axis=0)

for i in range(len(newError)):
    if(newError[i] != madeOneByOne[i]):
        state = False

if(state):
    print("it works !")
else:
    print("it don't works")