import pickle
import os
import numpy as np
import mlp as mlp
import CrossValidation as cv

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot(Y):
    N, C = Y.shape[0], (np.max(Y)+1)
    y_hot = np.zeros([N, C])
    y_hot[np.arange(N),Y.flatten()] = 1
    return y_hot


root_path_Jessica = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project3/cifar-10-batches-py/data_batch/'
root_path_Claire = '/Users/liuxijun/Downloads/Applied_Machine_Learning/Project3/cifar-10-batches-py/data_batch/'
data_dir = sorted(os.listdir(root_path_Jessica))  # <-------
dict_ls = []

# All train and test data are loaded as a list of dict
for i in range(len(data_dir)):
    final_path = os.path.join(root_path_Jessica, data_dir[i]) # <---------
    dict_ls.append(unpickle(final_path))




X_train = np.asarray(dict_ls[0][b'data'])
Y_train = np.asarray(dict_ls[0][b'labels'])[:, None]


X_test = np.asarray(dict_ls[5][b'data'])
Y_test = np.asarray(dict_ls[5][b'labels'])[:, None]
Y_test = one_hot(Y_test)

# concatenate all training data
for i in range(1,4):
    tempX = np.asarray(dict_ls[i][b'data'])
    tempY = np.asarray(dict_ls[i][b'labels'])[:, None]
    tempX = np.vstack((X_train,tempX))
    tempY = np.vstack((Y_train, tempY))
    X_train = tempX
    Y_train = tempY

Y_train = one_hot(Y_train)

M = 10          # number of hiddne units
lr = 0.1/10000  # learning rate
eps = 1e-9
max_iters = 15
batch_size = 500

N,D = X_train.shape
N,K = Y_train.shape
W = np.random.randn(M, K) * 0.01
V = np.random.randn(D, M) * 0.01

mlp_nn = mlp.MLP(W,V)
#%%
#mlp_nn.fit(X_train, Y_train, M, lr, max_iters, batch_size)
#
#
#Wh = mlp_nn.W
#Vh = mlp_nn.V
#
## test the model
#accuracy = mlp_nn.predict(X_test, Y_test, 'ReLu')

#%%
# 5-fold cross validation
cv_obj = cv.CrossValidation()
cv_accuracy, avg_cv_accuracy = cv_obj.cross_validation(X_train, Y_train, M, lr, max_iters, batch_size, 5, 'ReLu')

#folds_X, folds_Y = cv_obj.create_mini_batch(X_train, Y_train, 8000)
#X_test = folds_X[0]
#Y_test = folds_Y[0]
#X_train = np.concatenate(np.delete(folds_X,0,0), axis=0)
#Y_train = np.delete(folds_Y,0,0)

#%% check gradients

#def func(x):
#...     return x[0]**2 - 0.5 * x[1]**3
#>>> def grad(x):
#...     return [2 * x[0], -1.5 * x[1]**2]
#>>> from scipy.optimize import check_grad
#>>> check_grad(func, grad, [1.5, -1.5])
#
#from scipy.optimize import check_grad
#cost = mlp_nn.cost(X,Y,W,V)
#grad = mlp_nn.gradients(X,Y,W,V)
#error = check_grad(cost, grad, )

#%%
#temp = np.asarray(dict_ls[0][b'labels'])[:,None]
#
#temp1 = np.array([[1,1],[1,0],[9,0],[4,1],[2,1]])
#temp3 = np.delete(temp1,2,0)[:,0]
#temp2 = np.concatenate(np.delete(temp1,2,0), axis=0)
#temp2 = temp1[[0,1,2],[0,1]]
#ls = []
#temp3 = int(np.floor(1000/30))

#temp1[temp1>1] = 999
#print(temp1)
#temp = np.empty(temp1.shape)
#idx1 = np.where(temp1==1)
#print(idx1)
#print(temp1[idx1[0], idx1[1]])
#print(np.max(temp1, axis=-1)[:,None])

