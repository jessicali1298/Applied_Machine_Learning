#Your implementation should include the backpropagation and the mini-batch 
#gradient descent algorithm used (e.g., SGD). 
#You are encouraged to change the activation function (e.g., use ReLU), 
#and increase the number of layers, and play with the number of units per layer


# VARIABLES: type of activation function; number of layers, number of units/layer

import numpy as np

class mlp:
    def __init__(self, W, V, train_epoch_acc, valid_epoch_acc, test_epoch_acc):
        self.W = W
        self.V = V
        self.train_epoch_cc = train_epoch_acc
        self.valid_epoch_acc = valid_epoch_acc
        self.test_epoch_cc = test_epoch_acc
    
    def ReLu(self, z):
        zeroes = np.zeros(z.shape)
        return np.maximum(z,zeroes)
    
    def ReLuGrad(self, z):
        z[z<=0] = 0
        z[z>0] = 1
        return z
        
    def LeakyRelu(self, z):
        u = np.ones(z.shape)
        u[z<0] = np.double(0.01)
        z = z*u
        return z
    
    def LeakyReluGD(self, z):
        u = np.ones(z.shape)
        u[z<=0] = np.double(0.01)
        return u
    
    def SoftPlus (self, z):
        z = np.log(1+np.exp(z))
        return z
    
    def SoftPlusGD(self, z):
        exp_z = np.exp(z)
        u = exp_z /(1+exp_z)
        return u
    
    def logistic(self, z):
        # Z = N x M
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
    
    def one_hot(self, Y):
        N, C = Y.shape[0], (np.max(Y)+1)
        y_hot = np.zeros([N, C])
        y_hot[np.arange(N),Y.flatten()] = 1
        return y_hot
    
    
    def softmax(self,
                u # N x K
                ):
#        print('softmax input: ', u.shape)
        u_exp = np.exp(u - np.max(u,1)[:, None])
#        print('softmax - u_exp: ', u_exp)
        result = u_exp / np.sum(u_exp, axis=-1)[:, None]
        
#        print('softmax result: ', result)
#        print('new---------------------------------')
        return result
    
    
    def cost(self,
             X, #N x D
             Y, #N x K
             W, #M x K
             V  #D x M
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
            cols = mini_batch.shape[1] - 10
            mini_X = mini_batch[:, 0:cols]
            mini_Y = mini_batch[:, cols:]
            batch_ls.append((mini_X, mini_Y))
        
        # take care last mini_batch separately if batch_size not divisible
        if (all_data.shape[0] % batch_size != 0) :
            last_batch = all_data[num_batches*batch_size :, :]
            last_X = last_batch[:, 0:(cols - 10)]
            last_Y = last_batch[:, (cols - 10):-1]
            batch_ls.append((last_X, last_Y))
            
        return batch_ls
        
    # assume middle layer activation function is ReLu
    def gradients(self,
                  X, #N x D
                  Y, #N x K
                  W, #M x K 
                  V,  #D x M
                  act_func
                  ):
#        print('X: ', X.shape)
#        print('Y: ', Y.shape)
#        print('W: ', W.shape)
#        print('V: ', V.shape)
        
        if act_func == 'ReLu':
            Z = self.ReLu(np.dot(X,V)) #N x M     10000 x 10, hidden layer
            
            N,D = X.shape
            Yh = self.softmax(np.dot(Z,W)) #N x K     10000 x 10
            
            dY = Yh - Y     #N x K     10000 x 10
            
            dW = np.dot(Z.T, dY)/N  #M x K     10 x 10 
            
            dZ = np.dot(dY, W.T)    #N x M     10000 x 10
            
            hidden_grad = self.ReLuGrad(Z)
            
        elif act_func == 'Leaky_ReLu':
            Z = self.LeakyRelu(np.dot(X,V)) #N x M     10000 x 10, hidden layer
            
            N,D = X.shape
            Yh = self.softmax(np.dot(Z,W)) #N x K     10000 x 10
            
            dY = Yh - Y     #N x K     10000 x 10
            
            dW = np.dot(Z.T, dY)/N  #M x K     10 x 10 
            
            dZ = np.dot(dY, W.T)    #N x M     10000 x 10
            
            hidden_grad = self.LeakyReluGD(Z)
        elif act_func == 'Soft_Plus':
            Z = self.SoftPlus(np.dot(X,V)) #N x M     10000 x 10, hidden layer
            
            N,D = X.shape
            Yh = self.softmax(np.dot(Z,W)) #N x K     10000 x 10
            
            dY = Yh - Y     #N x K     10000 x 10
            
            dW = np.dot(Z.T, dY)/N  #M x K     10 x 10 
            
            dZ = np.dot(dY, W.T)    #N x M     10000 x 10
            
            hidden_grad = self.SoftPlusGD(Z)
           
        dV = np.dot(X.T, dZ * hidden_grad)/N

        return dW, dV
    
    
    
    def mini_GD(self, X, Y, X2, Y2, X3, Y3, M, lr, max_iters, batch_size, act_func):

        N,D = X.shape
        N,K = Y.shape
        W = np.random.randn(M, K) * 0.01
        V = np.random.randn(D, M) * 0.01
  
        train_epoch_ls = []
        valid_epoch_ls = []
        test_epoch_ls = []
        for i in range(max_iters):
            print('iteration: ', i)
            batches = self.create_mini_batch(X, Y, batch_size)
            t = 0
            for batch in batches:
#                print('batch number: ', t)
                mini_X = batch[0]
                mini_Y = batch[1]
                
                dW, dV = self.gradients(mini_X, mini_Y, W, V, act_func)

                W = W - lr*dW
                V = V - lr*dV
                self.W = W
                self.V = V
                t = t + 1
            predictions_train, train_epoch = self.predict(X, Y, act_func)
            predictions_valid, valid_epoch = self.predict(X2, Y2, act_func)
            predictions_test, test_epoch = self.predict(X3, Y3, act_func)
            train_epoch_ls.append(train_epoch)
            valid_epoch_ls.append(valid_epoch)
            test_epoch_ls.append(test_epoch)
            
        self.train_epoch_acc = train_epoch_ls
        self.valid_epoch_acc = valid_epoch_ls
        self.test_epoch_acc = test_epoch_ls
        return W, V
    
    
    def fit(self, X, Y, X2, Y2, X3, Y3, M, lr, max_iters, batch_size, act_func):
        W, V = self.mini_GD(X, Y, X2, Y2, X3, Y3, M, lr, max_iters, batch_size, act_func)
        self.W = W
        self.V = V
    
    
    def predict(self, X, Y, act_func):
        if (act_func == 'ReLu'): 
            softmax_result = self.softmax(np.dot(self.ReLu(np.dot(X, self.V)), self.W))
        elif (act_func == 'Leaky_ReLu'):
            softmax_result = self.softmax(np.dot(self.LeakyRelu(np.dot(X, self.V)), self.W))
        elif (act_func == 'Soft_Plus'):
            softmax_result = self.softmax(np.dot(self.SoftPlus(np.dot(X, self.V)), self.W))
        elif (act_func == 'Tanh'):
            # do something
            print('Using Tanh')
            
        elif(act_func == 'Sigmoid'):
            # do something
            print('Using Sigmoid')
        result = np.argmax(self.one_hot(np.argmax(softmax_result, axis=1)), axis=1)  
        Y_decode = np.argmax(Y, axis=1) 
        accuracy = np.mean(result == Y_decode)
        return result, accuracy
    
    def fit_earlyStopping(self, X, Y, X2, Y2, X3, Y3, M, lr, max_iters, batch_size, act_func):
        W, V = self.mini_GD_earlyStopping(self, X, Y, X2, Y2, X3, Y3, M, lr, max_iters, batch_size, act_func)
        self.W = W
        self.V = V
        
    def mini_GD_earlyStopping(self, X, Y, X2, Y2, X3, Y3, M, lr, max_iters, batch_size, act_func):

        N,D = X.shape
        N,K = Y.shape
        W = np.random.randn(M, K) * 0.01
        V = np.random.randn(D, M) * 0.01
  
        train_epoch_ls = []
        valid_epoch_ls = []
        test_epoch_ls = []
        max_test_ls = []
        max_test_epoch = 0
        itera = 0
        W_ls = []
        V_ls = []
        
        for i in range(max_iters):
            print('iteration: ', i)
            batches = self.create_mini_batch(X, Y, batch_size)
            t = 0
            for batch in batches:
#                print('batch number: ', t)
                mini_X = batch[0]
                mini_Y = batch[1]
                
                dW, dV = self.gradients(mini_X, mini_Y, W, V, act_func)

                W = W - lr*dW
                V = V - lr*dV
                self.W = W
                self.V = V
                t = t + 1
            predictions_train, train_epoch = self.predict(X, Y, act_func)
            predictions_valid, valid_epoch = self.predict(X2, Y2, act_func)
            predictions_test, test_epoch = self.predict(X3, Y3, act_func)
            train_epoch_ls.append(train_epoch)
            valid_epoch_ls.append(valid_epoch)
            test_epoch_ls.append(test_epoch)
            
            if test_epoch > max_test_epoch+0.005:
                max_test_epoch = test_epoch
                itera = i
            max_test_ls.append(max_test_epoch)
            W_ls.append(W)
            V_ls.append(V)
            if i - itera >= 4:
                return W_ls[itera], V_ls[itera]
            
        self.train_epoch_acc = train_epoch_ls
        self.valid_epoch_acc = valid_epoch_ls
        self.test_epoch_acc = test_epoch_ls
        return W, V
