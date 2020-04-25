import pickle
import os
import numpy as np
import mlp as mlp
import mlp_three as mlp_3
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
data_dir = sorted(os.listdir(root_path_Claire))  # <-------
dict_ls = []

# All train and test data are loaded as a list of dict
for i in range(len(data_dir)):
    final_path = os.path.join(root_path_Claire, data_dir[i]) # <---------
    dict_ls.append(unpickle(final_path))




X_train = np.asarray(dict_ls[0][b'data'])
Y_train = np.asarray(dict_ls[0][b'labels'])[:, None]

X_valid = np.asarray(dict_ls[4][b'data'])
Y_valid = np.asarray(dict_ls[4][b'labels'])[:, None]

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

M1 = 1000         # number of hidden units
M2 = 256

# lr = 0.1/6000  # learning rate
# eps = 1e-9
# max_iters = 2
# batch_size = 40

lr = 0.1/10000  # learning rate
max_iters = 40 #40
batch_size = 40 #40


N,D = X_train.shape
N,K = Y_train.shape
W = np.random.randn(M2, K) * 0.01
P = np.random.randn(M1, M2) * 0.01
V = np.random.randn(D, M1) * 0.01
train_acc_ls = []
valid_acc_ls = []
test_acc_ls = []
train_acc_ls3 = []
valid_acc_ls3 = []
test_acc_ls3 = []

#%%

mlp_nn = mlp.mlp(W,V,train_acc_ls, valid_acc_ls, test_acc_ls)

#mlp_nn.fit(X_train, Y_train, M1, lr, max_iters, batch_size, 'ReLu')
#mlp_nn.fit(X_train, Y_train, M1, lr, max_iters, batch_size, 'Leaky_ReLu')
mlp_nn.fit(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, 
           M1, lr, 40 ,40, 'Leaky_ReLu')
Wh = mlp_nn.W
Vh = mlp_nn.V

# test the model
#predictions, accuracy = mlp_nn.predict(X_test, Y_test, 'ReLu')
#predictions, accuracy = mlp_nn.predict(X_test, Y_test, 'Leaky_ReLu')
#predictions, accuracy = mlp_nn.predict(X_test, Y_test, 'Soft_Plus')


# can check train/test accuracy at each Epoch
train_epoch = mlp_nn.train_epoch_acc
test_epoch = mlp_nn.test_epoch_acc
valid_epoch = mlp_nn.valid_epoch_acc


#%%
accuracy_ls = []
for i in range(1):
    mlp_nn3 = mlp_3.mlp_three(W,P,V, train_acc_ls3, valid_acc_ls3, test_acc_ls3)
    mlp_nn3.fit_three(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, 
                      1024, 128, 0.1/650, max_iters, batch_size, 'Soft_Plus')
train_epoch3_soft = mlp_nn3.train_epoch_acc
valid_epoch3_soft = mlp_nn3.valid_epoch_acc
test_epoch3_soft = mlp_nn3.test_epoch_acc
