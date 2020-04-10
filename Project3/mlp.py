
#Your implementation should include the backpropagation and the mini-batch 
#gradient descent algorithm used (e.g., SGD). 
#You are encouraged to change the activation function (e.g., use ReLU), 
#and increase the number of layers, and play with the number of units per layer



# Questions:
#1. structure of MLP (numerous hidden layers + 1 activation layer?)
#2. 2-layer MLP is 2 hidden layers?
#3. Softmax likelihood

# VARIABLES: type of activation function; number of layers, number of units/layer

import numpy as np
def logsumexp(Z, # Z x K
              ):
    Zmax = np.max(Z, axis=1)[:,None]
    lse = Zmax + np.log(np.sum(np.exp(Z - Zmax),axis=1))[:,None]
    return lse #N

def softmax(
        u, # N x K
        ):
    u_exp = np.exp(u - np.max(u,1)[:, None])
    return u_exp / np.sum(u_exp, axis=-1)[:, None]

def cost(X, #N x D
         Y, #N x K
         W, #M x K
         V, #N x M
         ):
    Q = np.dot(X,V)
    Z = logistic (Q)
    U= np.dot(Z, W)
    Yh = softmax(U)
    nll = -np.mean(np.sum(U*Y, 1) - logsumexp(U))
    return nll

def gradients(X, #N x D
              Y, #N x K
              W, #M x K 
              V, #D x M
              ):
    Z = logistic(np.dot(X,V)) #N x M
    N,D = X.shape
    Yh = softmax(np.dot(Z,W)) #N x K
    dY = Yh - Y #N x K
    dW = np.dot(Z.T, dY)/N #M x K
    dZ = np.dot(dY, W.T) #N x M
    dV = np.dot(X.T, dZ * Z * (1-Z))/N #D x M
    return dW, dV


def GD(X, Y, M, lr=0.1, eps=1e-9, max_iters=100000):
    N,D = X.shape
    N,K = Y.shape
    W = np.random.randn(M, K) * 0.01
    V = np.random.randn(D,M) * 0.01
    dW = np.inf * np.ones_like(W)
    t = 0
    while np.linalg.norm(dW) > eps and t < max_iters:
        dW, dV = gradients(X, Y, W, V)
        W = W - lr*dW
        V = V - lr*dV
        t += 1
    return W, V
    







