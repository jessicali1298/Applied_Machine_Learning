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

class MLP:
    
    def ReLu(self, z):
        return np.max(z,0)
    
    def logistic(self, z):
        # Z = N xM
        pos_num = np.where(z>=0)
        neg_num = np.where(z<0)
        
        ans = np.empty(z.shape)
        
        ans[pos_num[0], pos_num[1]] = 1/(1+np.exp(-z[pos_num[0], pos_num[1]]))
        ans[neg_num[0], neg_num[1]] = np.exp(z[neg_num[0], neg_num[1]])/(1+np.exp(z[neg_num[0], neg_num[1]]))
    
        return ans
    
    def logsumexp(self,
                  Z # Z x K
                  ):
        Zmax = np.max(Z, axis=1)[:,None] # find Max by row & convert into Nx1 vecotr
        lse = Zmax + np.log(np.sum(np.exp(Z - Zmax),axis=1))[:,None]
        return lse #N
    
    def softmax(self,
                u # N x K
                ):
        u_exp = np.exp(u - np.max(u,1)[:, None])
        return u_exp / np.sum(u_exp, axis=-1)[:, None]
    
    def cost(self,
             X, #N x D
             Y, #N x K
             W, #M x K
             V #N x M
             ):
        Q = np.dot(X,V)
        Z = self.logistic(Q)
        U = np.dot(Z, W)
    #    Yh = softmax(U)
        nll = -np.mean(np.sum(U*Y, 1) - self.logsumexp(U))
        return nll
    
    # assume middle layer activation function is logistic sigmoid
    def gradients(self,
                  X, #N x D
                  Y, #N x K
                  W, #M x K 
                  V #D x M
                  ):
        print('X: ', X.shape)
        print('Y: ', Y.shape)
        print('W: ', W.shape)
        print('V: ', V.shape)
        
        Z = self.logistic(np.dot(X,V)) #N x M     10000 x 10
        print('input of sigmoid: ', Z.shape)
        
        N,D = X.shape
        Yh = self.softmax(np.dot(Z,W)) #N x K     10000 x 1
        print('Yh: ', Yh.shape)
        
        dY = Yh - Y     #N x K     10000 x 10000
        print('dY: ', dY.shape)
        
        dW = np.dot(Z.T, dY)/N  #M x K     10 x 10000
        print('dW: ', dW.shape)
        
        dZ = np.dot(dY, W.T)    #N x M
        print('dZ: ', dZ.shape)
        
        dV = np.dot(X.T, dZ * Z * (1-Z))/N #D x M
        print('dV: ', dV.shape)
        return dW, dV
    
    
    def GD(self, X, Y, M, lr, eps, max_iters):
        N,D = X.shape
        N,K = Y.shape
        W = np.random.randn(M, K) * 0.01
        V = np.random.randn(D, M) * 0.01
        dW = np.inf * np.ones_like(W)
        t = 0
        while np.linalg.norm(dW) > eps and t < max_iters:
            dW, dV = self.gradients(X, Y, W, V)
            W = W - lr*dW
            V = V - lr*dV
            t += 1
        return W, V
        
    
    
    
    
    
    
    
