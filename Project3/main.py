import pickle
import os
import numpy as np
import mlp as mlp

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project3/cifar-10-batches-py/data_batch/'
root_path = '/Users/liuxijun/Downloads/Applied_Machine_Learning/Project3/cifar-10-batches-py/data_batch/'
data_dir = sorted(os.listdir(root_path))
dict_ls = []

# All train and test data are loaded as a list of dict
for i in range(len(data_dir)):
    final_path = os.path.join(root_path, data_dir[i])
    dict_ls.append(unpickle(final_path))




X = np.asarray(dict_ls[0][b'data'])
Y = np.asarray(dict_ls[0][b'labels'])[:, None]
M = 10          # number of hiddne units
lr = 0.1        # learning rate
eps = 1e-9
max_iters = 1
batch_size = 5000

N,D = X.shape
N,K = Y.shape
W = np.random.randn(M, K) * 0.01
V = np.random.randn(D, M) * 0.01

mlp_nn = mlp.MLP(W,V)
mlp_nn.fit(X, Y, M, lr, eps, max_iters, batch_size)


Wh = mlp_nn.W
Vh = mlp_nn.V
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
temp1 = np.array([[1,1],[1,0],[1,0],[4,1],[2,1]])
temp2 = temp1[[0,1,2],[0,1]]
#ls = []
#temp3 = int(np.floor(1000/30))

#temp1[temp1>1] = 999
#print(temp1)
#temp = np.empty(temp1.shape)
#idx1 = np.where(temp1==1)
#print(idx1)
#print(temp1[idx1[0], idx1[1]])
#print(np.max(temp1, axis=-1)[:,None])

