#Your implementation should include the backpropagation and the mini-batch 
#gradient descent algorithm used (e.g., SGD). 
#You are encouraged to change the activation function (e.g., use ReLU), 
#and increase the number of layers, and play with the number of units per layer


# VARIABLES: type of activation function; number of layers, number of units/layer

import numpy as np

class MLP:
    def __init__(self, W, V):
        self.W = W
        self.V = V
    
    def ReLu(self, z):
        zeroes = np.zeros(z.shape)
        return np.maximum(z,zeroes)
    
    def ReLuGrad(self, z):
        z[z<=0] = 0
        z[z>0] = 1
        return z
        
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
        print('softmax input: ', u)
        u_exp = np.exp(u - np.max(u,1)[:, None])
        print('softmax - u_exp: ', u_exp)
        result = u_exp / np.sum(u_exp, axis=-1)[:, None]
        print('result: ', result)
        return result
    
    def cost(self,
             X, #N x D
             Y, #N x K
             W, #M x K
             V  #N x M
             ):
        Q = np.dot(X,V)
        Z = self.ReLu(Q)
        U = np.dot(Z, W)
    #    Yh = softmax(U)
        nll = -np.mean(np.sum(U*Y, 1) - self.logsumexp(U))
        return nll
    
    def create_mini_batch(self, X, Y, batch_size):
        all_data = np.hstack((X,Y))  # stack X and Y horizontally [[X1, Y1]...[Xn, Yn]]
        np.random.shuffle(all_data)  # shuffle data before creating batches
        num_batches = int(np.floor(all_data.shape[0] / batch_size))  # total number of mini_batches
        
        batch_ls = []
        
        # create mini_batches
        for i in range(num_batches):
            mini_batch = all_data[i*batch_size : (i+1)*batch_size, :]
            mini_X = mini_batch[:, 0:-1]
            mini_Y = mini_batch[:, -1]
            batch_ls.append((mini_X, mini_Y))
        
        # take care last mini_batch separately if batch_size not divisible
        if (all_data.shape[0] % batch_size != 0) :
            last_batch = all_data[num_batches*batch_size :, :]
            last_X = last_batch[:, 0:-1]
            last_Y = last_batch[:, -1]
            batch_ls.append((last_X, last_Y))
            
        return batch_ls
        
    # assume middle layer activation function is ReLu
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
        
        Z = self.ReLu(np.dot(X,V)) #N x M     10000 x 10, hidden layer
        print('input of ReLu: ', Z)
        
        N,D = X.shape
        Yh = self.softmax(np.dot(Z,W)) #N x K     10000 x 1
        print('Yh: ', Yh)
        
        dY = Yh - Y     #N x K     10000 x 1
        print('dY: ', dY)
        
        dW = np.dot(Z.T, dY)/N  #M x K     10 x 10000 
        print('dW: ', dW)
        
        dZ = np.dot(dY, W.T)    #N x M
        print('dZ: ', dZ)
        
#        dV = np.dot(X.T, dZ * Z * (1-Z))/N #D x M 
        dV = np.dot(X.T, dZ * self.ReLuGrad(Z))/N
        print('dV: ', dV)
        print('')
        return dW, dV
    
    
    def mini_GD(self, X, Y, M, lr, eps, max_iters, batch_size):
        N,D = X.shape
        N,K = Y.shape
        W = np.random.randn(M, K) * 0.01
        V = np.random.randn(D, M) * 0.01
        dW = np.inf * np.ones_like(W)

        for i in range(max_iters):
            batches = self.create_mini_batch(X, Y, batch_size)
            for batch in batches:
                mini_X = batch[0]
                mini_Y = batch[1][:,None]
                
                dW, dV = self.gradients(mini_X, mini_Y, W, V)
                W = W - lr*dW
                V = V - lr*dV
            
        return W, V
   
    
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
    
    
    def fit(self, X, Y, M, lr, eps, max_iters, batch_size):
        W, V = self.mini_GD(X, Y, M, lr, eps, max_iters, batch_size)
        self.W = W
        self.V = V
    
    
    def predict(self, X, Y, act_func):
        if (act_func == 'ReLu'):
            #g(W*h(Vx))
            result = self.softmax(np.dot(self.W, self.ReLu(np.dot(self.V,X))))
            
        elif (act_func == 'Tanh'):
            # do something
            print('Using Tanh')
            
        elif(act_func == 'Sigmoid'):
            # do something
            print('Using Sigmoid')
            
        accuracy = np.mean(result == Y)
        return accuracy
    
    
    
    
