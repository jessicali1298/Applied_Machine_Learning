import pickle
import os
import numpy as np
import MLP as mlp

mlp_nn = mlp.MLP()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

root_path = '/Users/j.li/School/U4_WINTER/COMP 551/Applied_Machine_Learning/Project3/cifar-10-batches-py/data_batch/'
data_dir = sorted(os.listdir(root_path))
dict_ls = []

# All train and test data are loaded as a list of dict
for i in range(len(data_dir)):
    final_path = os.path.join(root_path, data_dir[i])
    dict_ls.append(unpickle(final_path))

#print(dict_ls[0].keys())

W,V = mlp_nn.GD(np.asarray(dict_ls[0][b'data']), np.asarray(dict_ls[0][b'labels']), 
                10, 0.1, 1e-9, 100000)
GD(self, X, Y, M, lr=0.1, eps=1e-9, max_iters=100000):

#temp1 = np.array([[1,3,1],[1,4,2]])
#temp = np.empty(temp1.shape)
#idx1 = np.where(temp1==1)
#print(idx1)
#print(temp1[idx1[0], idx1[1]])
#print(np.max(temp1, axis=-1)[:,None])
