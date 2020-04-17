import numpy as np
import mlp as mlp

class CrossValidation:
    def create_mini_batch(self, X, Y, batch_size):
        all_data = np.hstack((X,Y))  # stack X and Y horizontally [[X1, Y1]...[Xn, Yn]]
        np.random.shuffle(all_data)  # shuffle data before creating batches
        num_batches = int(np.floor(all_data.shape[0] / batch_size))  # total number of mini_batches
        batch_X_ls = []
        batch_Y_ls = []
        
        # create mini_batches
        for i in range(num_batches):
            mini_batch = all_data[i*batch_size : (i+1)*batch_size, :]
            cols = mini_batch.shape[1] - 10
            mini_X = mini_batch[:, 0:cols]
            mini_Y = mini_batch[:, cols:]
            batch_X_ls.append(mini_X)
            batch_Y_ls.append(mini_Y)
        
        # take care last mini_batch separately if batch_size not divisible
        if (all_data.shape[0] % batch_size != 0) :
            last_batch = all_data[num_batches*batch_size :, :]
            last_X = last_batch[:, 0:(cols - 10)]
            last_Y = last_batch[:, (cols - 10):-1]
            batch_X_ls.append(last_X)
            batch_Y_ls.append(last_Y)
            
        return batch_X_ls, batch_Y_ls

    
    def cross_validation(self, X, Y, M, lr, max_iters, batch_size, k, act_func):
        fold_size = int(np.floor(X.shape[0]/k))
        accuracy_ls = []
        
         
        folds_X, folds_Y = self.create_mini_batch(X, Y, fold_size)
        folds_X = np.asarray(folds_X)
        folds_Y = np.asarray(folds_Y)
        
        for i in range(k):
            X_test = folds_X[i]
            Y_test = folds_Y[i]
            X_train = np.concatenate(np.delete(folds_X,i,0), axis=0)
            Y_train = np.concatenate(np.delete(folds_Y,i,0), axis=0)
            
            N,D = X_train.shape
            N,K = Y_train.shape
            W = np.random.randn(M, K) * 0.01
            V = np.random.randn(D, M) * 0.01
            mlp_nn = mlp.mlp(W,V)
            
            mlp_nn.fit(X_train, Y_train, M, lr, max_iters, batch_size, act_func)
            predictions, accuracy = mlp_nn.predict(X_test, Y_test, act_func)
            accuracy_ls.append(accuracy)
            
#        if (act_func == 'ReLu'):
#            
#        elif (act_func == 'Tanh'):
#            # do something
#            print('Using Tanh')
#            
#        elif(act_func == 'Sigmoid'):
#            # do something
#            print('Using Sigmoid')
    
        return accuracy_ls, np.mean(accuracy_ls)