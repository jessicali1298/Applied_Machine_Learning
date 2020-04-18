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

M1 = 1000          # number of hidden units
M2 = 200

# lr = 0.1/6000  # learning rate
# eps = 1e-9
# max_iters = 2
# batch_size = 40

lr = 0.1/800  # learning rate
max_iters = 40 #40
batch_size = 40 #40


N,D = X_train.shape
N,K = Y_train.shape
W = np.random.randn(M2, K) * 0.01
P = np.random.randn(M1, M2) * 0.01
V = np.random.randn(D, M1) * 0.01
train_acc_ls = []
test_acc_ls = []

#%%

mlp_nn = mlp.mlp(W,V)

#mlp_nn.fit(X_train, Y_train, M1, lr, max_iters, batch_size, 'ReLu')
#mlp_nn.fit(X_train, Y_train, M1, lr, max_iters, batch_size, 'Leaky_ReLu')
mlp_nn.fit(X_train, Y_train, M1, lr, max_iters, batch_size, 'Soft_Plus')
Wh = mlp_nn.W
Vh = mlp_nn.V

# test the model
#predictions, accuracy = mlp_nn.predict(X_test, Y_test, 'ReLu')
#predictions, accuracy = mlp_nn.predict(X_test, Y_test, 'Leaky_ReLu')
predictions, accuracy = mlp_nn.predict(X_test, Y_test, 'Soft_Plus')

#%%
# 5-fold cross validation
#cv_obj = cv.CrossValidation()
#cv_accuracy, avg_cv_accuracy = cv_obj.cross_validation(X_train, Y_train, M, lr, max_iters, batch_size, 5, 'ReLu')

#%%
# test 3-layer MLP
M1_ls = [10,30,50,80,100,150,200,300,500,800,1000,2000]
M2_ls = [10,30,50,80,100,150,200,300,500,800,1000]
lr_ls = [0.1/100, 0.1/300,0.1/500, 0.1/800, 0.1/1000, 0.1/3000, 0.1/7000, 0.1/10000]
max_iters_ls = [3, 5, 10, 20, 30, 50, 80, 100]
batch_size_ls = [20,40,80,100,200,400,800,1000,2000]

accuracy_ls = []
for i in range(1):
    mlp_nn3 = mlp_3.mlp_three(W,P,V, train_acc_ls, test_acc_ls)
    mlp_nn3.fit_three(X_train, Y_train, X_test, Y_test, M1, M2, lr, max_iters, batch_size, 'ReLu')
    
    # test the model
#    predictions, accuracy = mlp_nn3.predict_three(X_test, Y_test, 'ReLu')
#    accuracy_ls.append(accuracy)

train_epoch = mlp_nn3.train_epoch_acc
test_epoch = mlp_nn3.test_epoch_acc


#%%
M1_result = [0.3916, 0.4593, 0.4648, 0.4689, 0.4742, 0.4866, 0.4801, 0.4925, 0.4852, 0.4972, 0.4965, 0.5043]
lr_result = [0.4975, 0.5075, 0.4936, 0.4755, 0.4655, 0.4268, 0.3907, 0.3758]
max_iters_result = [0.384, 0.4016, 0.4391, 0.4587, 0.4809, 0.5, 0.4999, 0.5068]
batch_size_result = [0.5233, 0.505, 0.5042, 0.4926, 0.4586, 0.4492, 0.4078, 0.412, 0.3739]

#%%





#%%
#temp1 = np.argmax(np.array([[0,0,0,1,0,0], [1,0,0,0,0,0], [0,1,0,0,0,0]]),axis=1)
#temp2 = np.argmax(np.array([[0,0,0,1,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0]]), axis=1)
#temp3 = np.array([[0,0,0,1,0,0], [1,0,0,0,0,0], [0,1,0,0,0,0]])
#result = np.mean(temp1 == temp2)

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
#temp1 = np.array([[1,1],[1,0]])
#temp2 = np.array([[2,3],[4,1]])
#temp3 = temp1*temp2
#temp4 = np.dot(temp1, temp2)
#print(temp3 == temp4)

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